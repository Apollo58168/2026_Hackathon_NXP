#!/usr/bin/env python3
"""Logitech C270 -> MiDaS v2.1 Small -> HDMI depth preview on i.MX93."""
import argparse
import glob
import os
import signal
import time

import numpy as np

MODEL_GLOBS = (
    "/opt/gopoint-apps/downloads/*midas*vela*.tflite",
    "/opt/models/**/*midas*vela*.tflite",
    "./*midas*vela*.tflite",
)


def find_model(requested):
    if requested:
        if not os.path.isfile(requested):
            raise FileNotFoundError(requested)
        return requested
    for pattern in MODEL_GLOBS:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError("MiDaS Vela model not found; pass --model /path/to/*midas*vela*.tflite")


def normalize_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError("MiDaS returned no finite depth values")
    low, high = np.percentile(depth[finite], (2, 98))
    return np.clip((depth - low) / max(float(high - low), 1e-6), 0, 1)


def colorize(depth):
    """Small dependency-free blue->cyan->yellow->red relative-depth map."""
    x = normalize_depth(depth)
    stops = np.array([0.0, 0.33, 0.66, 1.0])
    colors = np.array([[20, 20, 120], [0, 210, 255], [255, 240, 0], [220, 0, 0]], dtype=np.float32)
    return np.stack([np.interp(x, stops, colors[:, channel]) for channel in range(3)], axis=-1).astype(np.uint8)


def quantize_input(rgb, details):
    dtype = details["dtype"]
    real = rgb.astype(np.float32) / 255.0
    if np.issubdtype(dtype, np.floating):
        return real[None].astype(dtype)
    scale, zero = details["quantization"]
    if not scale:
        raise ValueError("quantized input tensor has no quantization scale")
    limits = np.iinfo(dtype)
    return np.clip(np.rint(real / scale + zero), limits.min, limits.max).astype(dtype)[None]


def dequantize_output(value, details):
    value = np.asarray(value).squeeze()
    scale, zero = details["quantization"]
    return (value.astype(np.float32) - zero) * scale if scale else value.astype(np.float32)


def gst_modules():
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    return Gst


def make_interpreter(model, delegate, threads):
    import tflite_runtime.interpreter as tflite
    kwargs = {"model_path": model, "num_threads": threads}
    if delegate:
        kwargs["experimental_delegates"] = [tflite.load_delegate(delegate)]
    interpreter = tflite.Interpreter(**kwargs)
    interpreter.allocate_tensors()
    return interpreter


def run(args):
    Gst = gst_modules()
    model = find_model(args.model)
    interpreter = make_interpreter(model, args.delegate, args.threads)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    shape = input_details["shape"]
    if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
        raise ValueError("expected NHWC RGB input, got %s" % list(shape))
    height, width = int(shape[1]), int(shape[2])

    camera_desc = (
        "v4l2src device=%s ! video/x-raw,format=YUY2,width=%d,height=%d,framerate=%d/1 ! "
        "queue max-size-buffers=1 leaky=downstream ! imxvideoconvert_pxp ! "
        "video/x-raw,format=RGB16,width=%d,height=%d ! videoconvert ! "
        "video/x-raw,format=RGB ! appsink name=sink drop=true max-buffers=1 sync=false"
    ) % (args.camera, args.camera_width, args.camera_height, args.camera_fps, width, height)
    output_desc = (
        "appsrc name=src is-live=true block=false do-timestamp=true format=time "
        "caps=video/x-raw,format=RGB,width=%d,height=%d,framerate=0/1 ! "
        "queue max-size-buffers=1 leaky=downstream ! videoscale add-borders=true ! "
        "video/x-raw,width=%d,height=%d,pixel-aspect-ratio=1/1 ! "
        "textoverlay name=overlay valignment=top halignment=left font-desc=\"Sans 18\" "
        "shaded-background=true ! waylandsink fullscreen=true sync=false"
    ) % (width * 2, height, args.display_width, args.display_height)

    camera = Gst.parse_launch(camera_desc)
    output = Gst.parse_launch(output_desc)
    sink, src, overlay = camera.get_by_name("sink"), output.get_by_name("src"), output.get_by_name("overlay")
    for pipeline in (camera, output):
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer pipeline failed to start")

    running = True
    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print("model:", model)
    print("camera:", args.camera, "input:", list(shape), "backend:", "NPU" if args.delegate else "CPU")
    print("white/warm = nearer relative depth; Ctrl+C to stop")

    frames, started = 0, time.monotonic()
    try:
        while running:
            sample = sink.emit("try-pull-sample", Gst.SECOND)
            if sample is None:
                raise RuntimeError("no C270 frame received; check --camera and v4l2 caps")
            buffer = sample.get_buffer()
            ok, mapped = buffer.map(Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError("cannot map camera frame")
            try:
                rgb = np.frombuffer(mapped.data, np.uint8).reshape(height, width, 3).copy()
            finally:
                buffer.unmap(mapped)

            interpreter.set_tensor(input_details["index"], quantize_input(rgb, input_details))
            inference_started = time.monotonic()
            interpreter.invoke()
            latency_ms = (time.monotonic() - inference_started) * 1000
            depth = dequantize_output(interpreter.get_tensor(output_details["index"]), output_details)
            if depth.shape != (height, width):
                raise ValueError("unexpected MiDaS output shape %s" % (depth.shape,))
            combined = np.concatenate((rgb, colorize(depth)), axis=1)
            out = Gst.Buffer.new_allocate(None, combined.nbytes, None)
            out.fill(0, combined.tobytes())
            src.emit("push-buffer", out)
            frames += 1
            fps = frames / max(time.monotonic() - started, 1e-6)
            overlay.set_property("text", "C270 RGB | MiDaS relative depth (warm = near)\nNPU %.1f ms | %.1f FPS" % (latency_ms, fps))
    finally:
        src.emit("end-of-stream")
        camera.set_state(Gst.State.NULL)
        output.set_state(Gst.State.NULL)


def self_test():
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    image = colorize(depth)
    assert image.shape == (4, 4, 3) and image.dtype == np.uint8
    assert normalize_depth(np.ones((2, 2))).sum() == 0
    details = {"dtype": np.int8, "quantization": (1 / 255, -128)}
    quantized = quantize_input(np.array([[[0, 127, 255]]], dtype=np.uint8), details)
    assert quantized.tolist() == [[[[-128, -1, 127]]]]
    assert np.allclose(dequantize_output(quantized, details), np.array([[[0, 127 / 255, 1]]]), atol=1e-3)
    print("self-test OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="/dev/video2")
    parser.add_argument("--model")
    parser.add_argument("--delegate", default="/usr/lib/libethosu_delegate.so")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/0")
        os.environ.setdefault("WAYLAND_DISPLAY", "/run/wayland-0")
        run(args)


if __name__ == "__main__":
    main()
