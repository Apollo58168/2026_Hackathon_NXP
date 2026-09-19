"""Run realtime MiDaS v2.1 small TFLite depth-map inference from a camera."""

import argparse
import time

import cv2
import numpy as np

from MidasDepthEstimation.midasDepthEstimator import midasDepthEstimator


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
	parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
	parser.add_argument("--height", type=int, default=720, help="Requested camera height")
	parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the preview")
	return parser.parse_args()


def main():
	args = parse_args()
	depth_estimator = midasDepthEstimator()

	# CAP_ANY is portable: DirectShow is Windows-only and prevents macOS capture.
	camera = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
	camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
	camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
	if not camera.isOpened():
		raise RuntimeError(
			f"Could not open camera {args.camera}. Grant camera access to your terminal, "
			"or choose another device with --camera."
		)

	# AVFoundation cameras on macOS can return a few empty frames while they
	# initialize. Wait briefly so the realtime preview does not exit at startup.
	deadline = time.monotonic() + 5
	while True:
		ret, image = camera.read()
		if ret:
			break
		if time.monotonic() >= deadline:
			camera.release()
			raise RuntimeError("Could not read a frame from the camera after waiting 5 seconds.")
		time.sleep(0.05)

	cv2.namedWindow("MiDaS realtime depth map (press q to quit)", cv2.WINDOW_NORMAL)
	try:
		while True:
			if image is None:
				ret, image = camera.read()
			if not ret:
				continue
			if not args.no_mirror:
				image = cv2.flip(image, 1)

			color_depth = depth_estimator.estimateDepth(image)
			overlay = cv2.addWeighted(image, 0.65, color_depth, 0.55, 0)
			cv2.putText(
				overlay, f"{depth_estimator.fps} FPS", (16, 36),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,
			)
			preview = np.hstack((image, color_depth, overlay))
			cv2.imshow("MiDaS realtime depth map (press q to quit)", preview)
			if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
				break
			image = None
	finally:
		camera.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
