#!/usr/bin/env python3
"""Fullscreen GTK C270 preview with an application-provided mouse cursor."""
import signal

import gi
gi.require_version("Gst", "1.0")
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, GLib, Gst, Gtk

WIDTH, HEIGHT = 640, 480


def cursor_pixbuf():
    """Create a cursor locally because this image has no XCursor theme installed."""
    size = 24
    pixels = bytearray(size * size * 4)
    for y in range(20):
        for x in range(min(y // 2 + 1, 10)):
            edge = x == 0 or x == min(y // 2, 9) or y == 19
            offset = (y * size + x) * 4
            pixels[offset:offset + 4] = bytes((0, 0, 0, 255) if edge else (255, 255, 255, 255))
    return GdkPixbuf.Pixbuf.new_from_bytes(
        GLib.Bytes.new(bytes(pixels)), GdkPixbuf.Colorspace.RGB, True, 8, size, size, size * 4
    )


class CameraWindow(Gtk.Window):
    def __init__(self, camera):
        super().__init__(title="C270 Camera")
        self.frame = None
        self.pending = False
        self.area = Gtk.DrawingArea()
        self.area.connect("draw", self.on_draw)
        self.add(self.area)
        self.connect("destroy", self.quit)
        self.connect("key-press-event", self.on_key)
        self.connect("realize", self.on_realize)

        description = (
            "v4l2src device=%s ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
            "queue max-size-buffers=1 leaky=downstream ! imxvideoconvert_pxp ! "
            "video/x-raw,format=RGB16,width=640,height=480 ! videoconvert ! "
            "video/x-raw,format=RGB ! appsink name=sink emit-signals=true "
            "drop=true max-buffers=1 sync=false"
        ) % camera
        self.pipeline = Gst.parse_launch(description)
        self.pipeline.get_by_name("sink").connect("new-sample", self.on_sample)
        self.fullscreen()
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_realize(self, _widget):
        cursor = Gdk.Cursor.new_from_pixbuf(Gdk.Display.get_default(), cursor_pixbuf(), 1, 1)
        self.get_window().set_cursor(cursor)

    def on_sample(self, sink):
        sample = sink.emit("pull-sample")
        data = sample.get_buffer().extract_dup(0, WIDTH * HEIGHT * 3)
        self.frame = GdkPixbuf.Pixbuf.new_from_bytes(
            GLib.Bytes.new(data), GdkPixbuf.Colorspace.RGB, False, 8, WIDTH, HEIGHT, WIDTH * 3
        )
        if not self.pending:
            self.pending = True
            GLib.idle_add(self.refresh)
        return Gst.FlowReturn.OK

    def refresh(self):
        self.pending = False
        self.area.queue_draw()
        return GLib.SOURCE_REMOVE

    def on_draw(self, widget, context):
        if self.frame is None:
            return False
        width, height = widget.get_allocated_width(), widget.get_allocated_height()
        scale = min(width / WIDTH, height / HEIGHT)
        context.translate((width - WIDTH * scale) / 2, (height - HEIGHT * scale) / 2)
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
    args = parser.parse_args()
    Gst.init(None)
    window = CameraWindow(args.camera)
    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(window.close))
    window.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
