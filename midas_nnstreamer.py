#!/usr/bin/env python3
"""C270 -> MiDaS/Ethos-U preview plus voted changed-crop detection."""
import argparse
import json
import os
import signal
import time

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, GLib, Gst, Gtk

from depth_denoise import denoise_relative_depth

MODEL = "/opt/gopoint-apps/downloads/midas_v2_1_small_quant_vela.tflite"
DETECTOR_MODEL = "/opt/gopoint-apps/downloads/ssdlite_mobilenet_v2_coco_quant_uint8_float32_no_postprocess_vela.tflite"
DETECTOR_LABELS = "/opt/gopoint-apps/downloads/coco_labels_list.txt"
DETECTOR_PRIORS = "/opt/gopoint-apps/downloads/box_priors.txt"
ENABLED_CONFIG = "/root/enabled_classes.json"
SIZE = 256
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480


def cursor_pixbuf():
    pixels = bytearray(24 * 24 * 4)
    for y in range(20):
        for x in range(min(y // 2 + 1, 10)):
            edge = x == 0 or x == min(y // 2, 9) or y == 19
            offset = (y * 24 + x) * 4
            pixels[offset:offset + 4] = bytes((0, 0, 0, 255) if edge else (255, 255, 255, 255))
    return GdkPixbuf.Pixbuf.new_from_bytes(
        GLib.Bytes.new(bytes(pixels)), GdkPixbuf.Colorspace.RGB, True, 8, 24, 24, 96
    )


class DepthWindow(Gtk.Window):
    def __init__(self, camera, model, detector=None, detect_crop=None, detect_after=30,
                 transaction_probe=None, active_layer=None, bilateral_diameter=5,
                 bilateral_sigma=0.08):
        super().__init__(title="MiDaS Relative Depth + voted SSD-Lite")
        self.frame = self.latest = self.rgb = self.analysis_rgb = None
        self.pending = False
        self.frames = 0
        self.started = time.monotonic()
        self.previous_depth = None
        self.latest_depth = None
        self.depth_change = 0.0
        self.detector = detector
        self.detect_crop = detect_crop
        self.detect_after = detect_after
        self.transaction_probe = transaction_probe
        self.active_layer = active_layer if active_layer in (1, 2) else None
        self.inventory = {1: {}, 2: {}}
        self.status_event = None
        self.probe_result_seen = False
        self.storage = None
        self.calibrator = None
        self.calibration_saved = False
        self.bilateral_diameter = int(bilateral_diameter)
        self.bilateral_sigma = float(bilateral_sigma)
        try:
            from hardware_calibration import TwoLayerCalibrator
            from storage import Storage

            self.storage = Storage("/root/smartdrawer.db")
            self.calibrator = TwoLayerCalibrator(
                bilateral_diameter=self.bilateral_diameter,
                bilateral_sigma=self.bilateral_sigma,
            )
            self.load_inventory()
        except Exception as error:
            print("hardware state storage unavailable:", error, flush=True)
        calibrated = self.storage is not None and self.storage.calibration_count() == 2
        self.phase = "READY" if calibrated else "INITIALIZATION"
        self.init_started = calibrated
        self.phase_button = "NEXT PHASE" if calibrated else "START INIT"
        self.detection_done = detector is None or detect_crop is None

        self.area = Gtk.DrawingArea()
        self.area.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.area.connect("draw", self.on_draw)
        self.area.connect("button-press-event", self.on_button_press)
        self.add(self.area)
        self.connect("realize", self.on_realize)
        self.connect("destroy", self.quit)
        self.connect("key-press-event", self.on_key)
        self.fullscreen()

        pipeline = (
            "v4l2src device=%s ! "
            "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! tee name=cam "
            "cam. ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! "
            "video/x-raw,format=RGB ! appsink name=rgb_sink emit-signals=true drop=true max-buffers=1 sync=false "
            "cam. ! queue max-size-buffers=2 leaky=downstream ! imxvideoconvert_pxp ! "
            "video/x-raw,format=RGB16,width=256,height=256 ! tee name=analysis "
            "analysis. ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! "
            "video/x-raw,format=RGB ! appsink name=analysis_sink emit-signals=true drop=true max-buffers=1 sync=false "
            "analysis. ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! "
            "video/x-raw,format=RGB ! tensor_converter ! "
            "tensor_transform mode=arithmetic option=typecast:float32,div:255 ! "
            "tensor_filter latency=1 framework=tensorflow2-lite model=%s "
            "custom=Delegate:External,ExtDelegateLib:libethosu_delegate.so ! "
            "tensor_sink name=depth_sink"
        ) % (camera, model)
        self.pipeline = Gst.parse_launch(pipeline)
        self.pipeline.get_by_name("rgb_sink").connect("new-sample", self.on_rgb)
        self.pipeline.get_by_name("analysis_sink").connect("new-sample", self.on_analysis_rgb)
        self.pipeline.get_by_name("depth_sink").connect("new-data", self.on_depth)
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("NNStreamer pipeline failed to start")

    def on_realize(self, _widget):
        self.get_window().set_cursor(
            Gdk.Cursor.new_from_pixbuf(Gdk.Display.get_default(), cursor_pixbuf(), 1, 1)
        )

    def on_message(self, _bus, message):
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print("GStreamer error:", error, debug or "", flush=True)
            self.close()

    def on_rgb(self, sink):
        sample = sink.emit("pull-sample")
        self.rgb = np.frombuffer(
            sample.get_buffer().extract_dup(0, DISPLAY_WIDTH * DISPLAY_HEIGHT * 3), np.uint8
        ).reshape(DISPLAY_HEIGHT, DISPLAY_WIDTH, 3).copy()
        return Gst.FlowReturn.OK

    def on_analysis_rgb(self, sink):
        sample = sink.emit("pull-sample")
        self.analysis_rgb = np.frombuffer(
            sample.get_buffer().extract_dup(0, SIZE * SIZE * 3), np.uint8
        ).reshape(SIZE, SIZE, 3).copy()
        return Gst.FlowReturn.OK

    def on_depth(self, _sink, buffer):
        depth = np.frombuffer(buffer.extract_dup(0, buffer.get_size()), np.float32)
        if depth.size != SIZE * SIZE:
            print("Unexpected MiDaS output size:", depth.size, flush=True)
            return
        depth = depth.reshape(SIZE, SIZE)
        self.latest_depth = depth.copy()
        finite = np.isfinite(depth)
        if not finite.any():
            print("MiDaS returned no finite depth", flush=True)
            return
        try:
            normalized_depth = denoise_relative_depth(
                depth, self.bilateral_diameter, self.bilateral_sigma
            )
        except ValueError as error:
            print("MiDaS depth normalization failed:", error, flush=True)
            return
        low, high = (float(value) for value in np.percentile(depth[finite], (2, 98)))
        if self.previous_depth is not None:
            self.depth_change = float(np.mean(np.abs(normalized_depth - self.previous_depth)))
        self.previous_depth = normalized_depth.copy()
        gray = (255 * normalized_depth).astype(np.uint8)
        gray = cv2.resize(gray, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_CUBIC)
        depth_rgb = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
        if self.rgb is None:
            return
        camera_rgb = self.rgb.copy()
        if self.calibrator is not None and self.calibrator.active:
            self.calibrator.feed(depth)
            self.sync_calibration_button()
            if self.calibrator.completed and not self.calibration_saved:
                self.finish_calibration()
        if self.transaction_probe is not None and self.analysis_rgb is not None:
            result = self.transaction_probe.feed(depth, self.analysis_rgb.copy())
            if result is not None and not self.probe_result_seen:
                self.probe_result_seen = True
                self.apply_transaction_result(result)
        if not self.detection_done and self.frames >= self.detect_after and self.analysis_rgb is not None:
            self.run_detector(camera_rgb, self.analysis_rgb.copy())
        overlay = cv2.addWeighted(camera_rgb, 0.7, depth_rgb, 0.6, 0)
        self.frames += 1
        fps = self.frames / max(time.monotonic() - self.started, 1e-6)
        stage = self.stage_text()
        self.annotate(depth_rgb, "MiDaS relative depth | warm = near", fps, low, high)
        self.annotate(overlay, "Camera + depth overlay", fps, low, high)
        self.draw_status_hud(overlay, stage)
        combined = np.hstack((depth_rgb, overlay))
        self.latest = combined.tobytes()
        if self.frames == 30:
            cv2.imwrite("/root/midas-hdmi-screenshot.png", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print("Screenshot saved: /root/midas-hdmi-screenshot.png", flush=True)
        if self.frames % 30 == 0:
            print(
                "frames=%d fps=%.1f depth=[%.2f, %.2f] std=%.2f delta=%.4f stage=%s"
                % (self.frames, fps, low, high, float(depth.std()), self.depth_change, stage),
                flush=True,
            )
        if not self.pending:
            self.pending = True
            GLib.idle_add(self.refresh)

    def stage_text(self):
        if self.status_event is not None:
            return self.status_event
        if self.calibrator is not None and self.calibrator.active:
            return self.calibrator.message
        if self.calibrator is not None and (
                self.calibrator.state == "error" or self.calibrator.state.startswith("ready_")):
            return self.calibrator.message
        if self.phase == "PUT":
            return "PUT"
        if self.phase == "TAKE":
            return "TAKE"
        if self.phase == "READY":
            return "READY"
        if self.init_started:
            return "INITIALIZING"
        if self.transaction_probe is not None:
            return {
                "capturing_a": "INITIALIZING: waiting for stable depth A",
                "waiting_item_motion": "READY: waiting for item motion",
                "capturing_b": "PROCESSING: waiting for stable depth B",
                "done": "PROCESSING COMPLETE",
            }.get(self.transaction_probe.state, self.transaction_probe.state.upper())
        if self.detector is not None and not self.detection_done:
            return "DETECTING OBJECT"
        return "INITIALIZATION"

    def load_inventory(self):
        if self.storage is None:
            return
        self.inventory = {1: {}, 2: {}}
        for row in self.storage.inventory_rows():
            layer = int(row["layer_no"])
            if layer in self.inventory:
                self.inventory[layer][str(row["canonical_label"])] = int(row["quantity"])

    def finish_calibration(self):
        if self.storage is not None:
            self.storage.finish_initialization(self.calibrator.calibrations)
            self.storage.set_metadata("depth_calibration_state", "ready")
            self.storage.set_metadata("near_is_positive", "1")
            if self.calibrator.closed_depth is not None:
                import base64
                import zlib

                payload = zlib.compress(self.calibrator.closed_depth.astype(np.float32).tobytes())
                self.storage.set_metadata("closed_depth_shape", "%d,%d" % self.calibrator.closed_depth.shape)
                self.storage.set_metadata("closed_depth_f32_zlib_b64", base64.b64encode(payload).decode("ascii"))
                self.storage.set_metadata("depth_noise", str(self.calibrator.noise))
            self.load_inventory()
        self.calibration_saved = True
        self.phase = "READY"
        self.init_started = True
        self.phase_button = "NEXT PHASE"
        self.status_event = self.calibrator.message

    def sync_calibration_button(self):
        labels = {
            "capturing_closed": "CAPTURING...",
            "waiting_open_1": "WAIT LAYER 1",
            "capturing_layer_1": "WAIT LAYER 1",
            "waiting_open_2": "WAIT LAYER 2",
            "capturing_layer_2": "WAIT LAYER 2",
            "ready_layer_1": "LAYER 1 INIT",
            "ready_layer_2": "LAYER 2 INIT",
            "ready_finish": "FINISH INIT",
            "complete": "NEXT PHASE",
            "error": "RESTART INIT",
        }
        if self.calibrator is not None and self.calibrator.state in labels:
            self.phase_button = labels[self.calibrator.state]

    def advance_phase(self):
        if self.calibrator is None:
            self.status_event = "CALIBRATION MODULE UNAVAILABLE"
            self.area.queue_draw()
            return
        if self.calibrator.active:
            return
        self.status_event = None
        state = self.calibrator.state
        if state in ("idle", "error") and self.phase == "INITIALIZATION":
            if self.storage is not None:
                self.storage.clear_for_initialization()
                self.inventory = {1: {}, 2: {}}
            self.calibrator.start()
            self.calibration_saved = False
            self.init_started = True
        elif state == "ready_layer_1":
            if not self.calibrator.begin_layer(1, self.latest_depth):
                self.status_event = self.calibrator.message
        elif state == "ready_layer_2":
            if not self.calibrator.begin_layer(2, self.latest_depth):
                self.status_event = self.calibrator.message
        elif state == "ready_finish":
            if self.calibrator.finish(self.latest_depth):
                self.finish_calibration()
            else:
                self.status_event = self.calibrator.message
        elif state in ("idle", "complete") and self.phase in ("INITIALIZATION", "READY"):
            self.phase = "PUT"
        elif self.phase == "PUT":
            self.phase = "TAKE"
        self.sync_calibration_button()
        if self.phase == "TAKE":
            self.phase_button = "DONE"
        self.area.queue_draw()

    def on_button_press(self, _widget, event):
        if event.button != 1 or not hasattr(self, "button_rect"):
            return False
        if self.calibrator is not None and self.calibrator.active:
            return True
        x, y, width, height = self.button_rect
        if x <= event.x <= x + width and y <= event.y <= y + height:
            self.advance_phase()
            return True
        return False

    def draw_phase_button(self, context, width, height):
        button_width, button_height = 210, 64
        x = width - button_width - 28
        y = height - button_height - 28
        self.button_rect = (x, y, button_width, button_height)
        context.set_source_rgb(0.02, 0.12, 0.18)
        context.rectangle(x, y, button_width, button_height)
        context.fill_preserve()
        context.set_source_rgb(0.0, 0.85, 0.85)
        context.set_line_width(3)
        context.stroke()
        context.set_source_rgb(1.0, 1.0, 1.0)
        context.select_font_face("Sans", 0, 1)
        context.set_font_size(22)
        extents = context.text_extents(self.phase_button)
        context.move_to(x + (button_width - extents.width) / 2 - extents.x_bearing,
                        y + (button_height - extents.height) / 2 - extents.y_bearing)
        context.show_text(self.phase_button)

    def apply_transaction_result(self, result):
        if not result.get("ok"):
            self.status_event = "CHANGE REJECTED: %s" % str(result.get("reason", "unknown")).upper()
            return
        detections = result.get("detections", [])
        if len(detections) != 1:
            self.status_event = "ITEM UNKNOWN: %d DETECTIONS" % len(detections)
            return
        detection = detections[0]
        action = str(result["action"]).upper()
        if self.active_layer is None:
            self.status_event = "%s %s: LAYER UNKNOWN" % (action, detection.label.upper())
            return
        items = self.inventory[self.active_layer]
        current = items.get(detection.label, 0)
        if self.storage is not None and self.storage.calibration_count() == 2:
            try:
                from types import SimpleNamespace
                from core import Detection, make_candidate

                change = SimpleNamespace(
                    action=action.lower(),
                    signed_change=float(result["signed_change"]),
                    crop=result["crop"],
                )
                candidate = make_candidate(
                    change,
                    Detection(detection.class_id, detection.label, detection.confidence),
                    self.active_layer,
                )
                self.storage.commit_candidate(candidate)
                self.load_inventory()
                items = self.inventory[self.active_layer]
                if action == "TAKE" and current <= 0:
                    self.status_event = "TAKE %s: UNTRACKED" % detection.label.upper()
                    return
                self.status_event = "%s %s %s LAYER %d" % (
                    action,
                    detection.label.upper(),
                    "->" if action == "PUT" else "<-",
                    self.active_layer,
                )
                return
            except Exception as error:
                self.status_event = "INVENTORY ERROR: %s" % error
                return
        if action == "PUT":
            items[detection.label] = current + 1
            self.status_event = "PUT %s -> LAYER %d" % (detection.label.upper(), self.active_layer)
        elif action == "TAKE":
            if current <= 0:
                self.status_event = "TAKE %s: UNTRACKED" % detection.label.upper()
            else:
                items[detection.label] = current - 1
                if items[detection.label] == 0:
                    del items[detection.label]
                self.status_event = "TAKE %s <- LAYER %d" % (detection.label.upper(), self.active_layer)

    def draw_status_hud(self, image, stage):
        lines = ["SMART DRAWER", "STATUS: %s" % stage]
        for layer in (1, 2):
            items = self.inventory[layer]
            contents = ", ".join("%s x%d" % (label, quantity)
                                  for label, quantity in sorted(items.items())
                                  if quantity > 0) or "empty"
            lines.append("LAYER %d: %s" % (layer, contents))
        line_height = 22
        box_width = 300
        box_height = 12 + line_height * len(lines)
        x0 = DISPLAY_WIDTH - box_width - 10
        y0 = 8
        cv2.rectangle(image, (x0, y0), (DISPLAY_WIDTH - 8, y0 + box_height), (0, 0, 0), -1)
        cv2.rectangle(image, (x0, y0), (DISPLAY_WIDTH - 8, y0 + box_height), (0, 255, 255), 1)
        for index, line in enumerate(lines):
            cv2.putText(image, line[:42], (x0 + 8, y0 + 19 + index * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42 if index else 0.48,
                        (255, 255, 255) if index else (0, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def annotate(image, title, fps, low, high):
        cv2.rectangle(image, (0, 0), (DISPLAY_WIDTH, 58), (0, 0, 0), -1)
        cv2.putText(image, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "FPS %.1f | range %.1f..%.1f" % (fps, low, high),
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 255), 1, cv2.LINE_AA)

    def run_detector(self, camera_rgb, analysis_rgb):
        self.detection_done = True
        x, y, width, height = self.detect_crop
        if width <= 0 or height <= 0 or x < 0 or y < 0:
            print("invalid --detect-crop; expected X Y W H", flush=True)
            return
        crop = analysis_rgb[y:y + height, x:x + width]
        if crop.shape[0] != height or crop.shape[1] != width:
            print("--detect-crop is outside the 256x256 analysis frame", flush=True)
            return
        started = time.monotonic()
        detections = self.detector.detect(crop)
        latency_ms = (time.monotonic() - started) * 1000.0
        preview_x = DISPLAY_WIDTH / float(SIZE)
        preview_y = DISPLAY_HEIGHT / float(SIZE)
        for detection in detections:
            bx, by, bw, bh = detection.box
            left = int((x + bx) * preview_x)
            top = int((y + by) * preview_y)
            right = int((x + bx + bw) * preview_x)
            bottom = int((y + by + bh) * preview_y)
            cv2.rectangle(camera_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                camera_rgb,
                "%s %.2f" % (detection.label, detection.confidence),
                (left, max(16, top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if detections:
            self.status_event = "DETECTED: " + ", ".join(item.label.upper() for item in detections)
        else:
            self.status_event = "DETECTOR: NO ITEM"
        print(
            "detector=ssdlite backend=%s calls=%d latency_ms=%.1f crop=%s detections=%s"
            % (self.detector.backend, self.detector.calls, latency_ms, self.detect_crop, detections),
            flush=True,
        )

    def refresh(self):
        self.pending = False
        if self.latest is not None:
            self.frame = GdkPixbuf.Pixbuf.new_from_bytes(
                GLib.Bytes.new(self.latest), GdkPixbuf.Colorspace.RGB, False, 8,
                DISPLAY_WIDTH * 2, DISPLAY_HEIGHT, DISPLAY_WIDTH * 6,
            )
            self.area.queue_draw()
        return GLib.SOURCE_REMOVE

    def on_draw(self, widget, context):
        context.set_source_rgb(0, 0, 0)
        context.paint()
        width, height = widget.get_allocated_width(), widget.get_allocated_height()
        if self.frame is not None:
            source_width = DISPLAY_WIDTH * 2
            scale = min((width - 48) / source_width, (height - 48) / DISPLAY_HEIGHT)
            context.translate((width - source_width * scale) / 2, (height - DISPLAY_HEIGHT * scale) / 2)
            context.scale(scale, scale)
            Gdk.cairo_set_source_pixbuf(context, self.frame, 0, 0)
            context.paint()
            context.identity_matrix()
        self.draw_phase_button(context, width, height)
        return False

    def on_key(self, _widget, event):
        if event.keyval in (Gdk.KEY_s, Gdk.KEY_S) and self.latest is not None:
            image = np.frombuffer(self.latest, np.uint8).reshape(DISPLAY_HEIGHT, DISPLAY_WIDTH * 2, 3)
            path = "/root/midas-depth-screenshot.png"
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print("Screenshot saved:", path, flush=True)
        elif event.keyval in (Gdk.KEY_Escape, Gdk.KEY_q, Gdk.KEY_Q):
            self.close()
        return False

    def quit(self, *_args):
        self.pipeline.set_state(Gst.State.NULL)
        Gtk.main_quit()


def load_enabled_labels(path):
    from ssdlite_detector import DEFAULT_ENABLED_LABELS

    if not path or not os.path.isfile(path):
        return DEFAULT_ENABLED_LABELS
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    return frozenset(item["canonical_label"] for item in data["classes"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="/dev/video2")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--detector-model", default=DETECTOR_MODEL)
    parser.add_argument("--detector-labels", default=DETECTOR_LABELS)
    parser.add_argument("--detector-priors", default=DETECTOR_PRIORS)
    parser.add_argument("--enabled-config", default=ENABLED_CONFIG)
    parser.add_argument("--detect-crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="run one detector inference on this RGB crop after depth starts")
    parser.add_argument("--detect-after", type=int, default=30,
                        help="depth frames before the one-shot crop inference (default: 30)")
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--layer", type=int, choices=(1, 2),
                        help="active drawer layer for transaction inventory updates")
    parser.add_argument("--transaction-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="capture depth Snapshot A/B in this 256x256 ROI and vote on its changed crop")
    parser.add_argument("--motion-threshold", type=float, default=0.04)
    parser.add_argument("--stability-threshold", type=float, default=0.015)
    parser.add_argument("--object-change-threshold", type=float, default=0.10)
    parser.add_argument("--direction-threshold", type=float, default=0.05)
    parser.add_argument("--bilateral-diameter", type=int, default=5)
    parser.add_argument("--bilateral-sigma", type=float, default=0.08)
    parser.add_argument("--noise-multiplier", type=float, default=1.5)
    parser.add_argument("--noise-warmup-frames", type=int, default=8)
    parser.add_argument("--change-noise-multiplier", type=float, default=4.0)
    parser.add_argument("--detector-frames", type=int, default=3,
                        help="stable crop frames used for detector consensus")
    parser.add_argument("--detector-vote-ratio", type=float, default=2 / 3)
    parser.add_argument("--detector-match-iou", type=float, default=0.40)
    args = parser.parse_args()
    if not os.path.isfile(args.model):
        raise FileNotFoundError(args.model)
    detector = None
    if args.detect_crop is not None or args.transaction_roi is not None:
        from ssdlite_detector import SSDLiteDetector

        detector = SSDLiteDetector(
            args.detector_model,
            args.detector_labels,
            args.detector_priors,
            confidence=args.confidence,
            enabled_labels=load_enabled_labels(args.enabled_config),
        )
    transaction_probe = None
    if args.transaction_roi is not None:
        from transaction_probe import DepthTransactionProbe

        transaction_probe = DepthTransactionProbe(
            tuple(args.transaction_roi),
            detector,
            motion_threshold=args.motion_threshold,
            stability_threshold=args.stability_threshold,
            object_change_threshold=args.object_change_threshold,
            direction_threshold=args.direction_threshold,
            bilateral_diameter=args.bilateral_diameter,
            bilateral_sigma=args.bilateral_sigma,
            noise_multiplier=args.noise_multiplier,
            noise_warmup_frames=args.noise_warmup_frames,
            change_noise_multiplier=args.change_noise_multiplier,
            detector_frames=args.detector_frames,
            detector_vote_ratio=args.detector_vote_ratio,
            detector_match_iou=args.detector_match_iou,
        )
    os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/0")
    os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
    Gst.init(None)
    window = DepthWindow(
        args.camera,
        args.model,
        detector,
        args.detect_crop,
        args.detect_after,
        transaction_probe,
        args.layer,
        bilateral_diameter=args.bilateral_diameter,
        bilateral_sigma=args.bilateral_sigma,
    )
    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(window.close))
    window.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
