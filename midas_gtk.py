#!/usr/bin/env python3
"""Fullscreen per-pixel MiDaS relative-depth preview for the C270."""
import os
import signal
import time

import gi
gi.require_version("Gst", "1.0")
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, GLib, Gst, Gtk
import cv2
import numpy as np

from camera_gtk import cursor_pixbuf
from midas_hdmi_demo import dequantize_output, find_model, make_interpreter

FRAME_WIDTH, FRAME_HEIGHT = 640, 360


class MidasWindow(Gtk.Window):
    def __init__(self, camera, model, delegate):
        super().__init__(title="C270 MiDaS Depth")
        self.interpreter = make_interpreter(model, delegate, 2)
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        shape = self.input_details["shape"]
        self.height, self.width = int(shape[1]), int(shape[2])
        self.frame = None
        self.latest_bytes = None
        self.pending = False
        self.frames = 0
        self.started = time.monotonic()
        self.previous_rgb = None
        self.previous_depth = None
        self.previous_display = None

        self.area = Gtk.DrawingArea()
        overlay = Gtk.Overlay()
        overlay.add(self.area)
        self.add(overlay)
        self.area.connect("draw", self.on_draw)
        self.connect("destroy", self.quit)
        self.connect("key-press-event", self.on_key)
        self.connect("realize", self.on_realize)

        pipeline = (
            "v4l2src device=%s ! video/x-raw,format=YUY2,width=640,height=360,framerate=30/1 ! "
            "queue max-size-buffers=1 leaky=downstream ! imxvideoconvert_pxp ! "
            "video/x-raw,format=RGB16,width=640,height=360 ! videoconvert ! "
            "video/x-raw,format=RGB ! appsink name=sink emit-signals=true "
            "drop=true max-buffers=1 sync=false"
        ) % camera
        self.pipeline = Gst.parse_launch(pipeline)
        self.pipeline.get_by_name("sink").connect("new-sample", self.on_sample)
        self.fullscreen()
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("camera pipeline failed")

    def on_realize(self, _widget):
        cursor = Gdk.Cursor.new_from_pixbuf(Gdk.Display.get_default(), cursor_pixbuf(), 1, 1)
        self.get_window().set_cursor(cursor)

    def on_sample(self, sink):
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            rgb = np.frombuffer(mapped.data, np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH, 3).copy()
        finally:
            buffer.unmap(mapped)
        model_rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        # Match the referenced MiDaS v2.1 Small preprocessing contract.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = ((model_rgb.astype(np.float32) / 255.0 - mean) / std)[None]
        self.interpreter.set_tensor(self.input_details["index"], tensor.astype(self.input_details["dtype"]))
        inference_started = time.monotonic()
        self.interpreter.invoke()
        latency = (time.monotonic() - inference_started) * 1000
        depth = dequantize_output(self.interpreter.get_tensor(self.output_details["index"]), self.output_details)

        depth_min, depth_max = float(depth.min()), float(depth.max())
        disparity = (255 * (depth - depth_min) / max(depth_max - depth_min, 1e-6)).astype(np.uint8)
        disparity = cv2.resize(disparity, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC)
        magma = cv2.cvtColor(cv2.applyColorMap(disparity, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)

        # The requested output is only the per-pixel MAGMA relative-depth map.
        self.latest_bytes = magma.tobytes()
        self.frames += 1
        fps = self.frames / (time.monotonic() - self.started)
        rgb_change = float(np.mean(np.abs(model_rgb.astype(np.int16) - self.previous_rgb.astype(np.int16)))) if self.previous_rgb is not None else 0.0
        depth_change = float(np.mean(np.abs(depth - self.previous_depth))) if self.previous_depth is not None else 0.0
        display_change = float(np.mean(np.abs(magma.astype(np.int16) - self.previous_display.astype(np.int16)))) if self.previous_display is not None else 0.0
        self.previous_rgb, self.previous_depth, self.previous_display = model_rgb, depth.copy(), magma.copy()
        if self.frames % 30 == 0:
            print("frames=%d fps=%.1f latency_ms=%.1f rgb_change=%.2f depth_change=%.4f display_change=%.2f" % (
                self.frames, fps, latency, rgb_change, depth_change, display_change), flush=True)
        if not self.pending:
            self.pending = True
            GLib.idle_add(self.refresh, latency, fps, rgb_change, depth_change)
        return Gst.FlowReturn.OK

    def refresh(self, latency, fps, rgb_change, depth_change):
        self.pending = False
        data = self.latest_bytes
        if data is not None:
            self.frame = GdkPixbuf.Pixbuf.new_from_bytes(
                GLib.Bytes.new(data), GdkPixbuf.Colorspace.RGB, False, 8,
                FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH * 3,
            )
        self.area.queue_draw()
        return GLib.SOURCE_REMOVE

    def on_draw(self, widget, context):
        if self.frame is None:
            return False
        width, height = widget.get_allocated_width(), widget.get_allocated_height()
        context.set_source_rgb(0, 0, 0)
        context.paint()
        source_width, source_height = FRAME_WIDTH, FRAME_HEIGHT
        padding = 24
        scale = min((width - padding * 2) / source_width, (height - padding * 2) / source_height)
        context.translate((width - source_width * scale) / 2, (height - source_height * scale) / 2)
        context.scale(scale, scale)
        Gdk.cairo_set_source_pixbuf(context, self.frame, 0, 0)
        context.paint()
        return False

    def on_key(self, _widget, event):
        if event.keyval in (Gdk.KEY_Escape, Gdk.KEY_q, Gdk.KEY_Q):
            self.close()
        return False

    def quit(self, *_args):
        self.pipeline.set_state(Gst.State.NULL)
        Gtk.main_quit()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="/dev/video2")
    parser.add_argument("--model")
    parser.add_argument("--delegate", default="/usr/lib/libethosu_delegate.so")
    args = parser.parse_args()
    os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/0")
    os.environ.setdefault("WAYLAND_DISPLAY", "/run/wayland-0")
    Gst.init(None)
    window = MidasWindow(args.camera, find_model(args.model), args.delegate)
    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(window.close))
    window.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
