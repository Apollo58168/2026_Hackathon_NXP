#!/usr/bin/env python3
"""Submit Metric Small ONNX to eIQ: ONNX2Quant then TFLiteConversion.

Run this only on a host that can reach the eIQ AI Toolkit backend.  The script
uses the documented REST endpoints and deliberately requires --execute before
creating models/datasets or starting optimization jobs.
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path
from typing import Any

import requests


DEFAULT_ONNX = Path("models/depth_anything_v2_metric_hypersim_vits_518.onnx")
DEFAULT_TFLITE = Path("artifacts/depth_anything_v2_metric_hypersim_vits_518_int8.tflite")
INPUT_NAME = "image"


class EiqError(RuntimeError):
    pass


def response_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        raise EiqError(f"{response.request.method} {response.url}: non-JSON response: {response.text[:500]}") from error
    if not response.ok:
        raise EiqError(f"{response.request.method} {response.url}: HTTP {response.status_code}: {payload}")
    return payload


def validate_calibration_zip(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        samples = [name for name in archive.namelist() if name.startswith(f"{INPUT_NAME}/") and name.endswith(".npy")]
    if not samples:
        raise EiqError(f"{path} must contain {INPUT_NAME}/*.npy for the ONNX input node")
    return len(samples)


class EiqClient:
    def __init__(self, backend: str, *, timeout: float, poll_seconds: float) -> None:
        self.backend = backend.rstrip("/")
        self.timeout = timeout
        self.poll_seconds = poll_seconds
        self.session = requests.Session()

    def _get(self, path: str) -> dict[str, Any]:
        return response_json(self.session.get(f"{self.backend}{path}", timeout=self.timeout))

    def check_backend(self) -> list[str]:
        payload = self._get("/optimizations/passes")
        passes = [item["type"] for item in payload["data"]["passes"]]
        required = {"ONNX2Quant", "TFLiteConversion"}
        missing = required - set(passes)
        if missing:
            raise EiqError(f"backend is missing required passes: {', '.join(sorted(missing))}")
        return passes

    def upload_model(self, path: Path) -> str:
        create = response_json(self.session.post(f"{self.backend}/models", json={"model_type": "onnx"}, timeout=self.timeout))
        model_uuid = str(create["data"]["model"]["uuid"])
        with path.open("rb") as model_file:
            upload = self.session.post(
                f"{self.backend}/models/{model_uuid}",
                files={"model_file": (path.name, model_file, "application/octet-stream")},
                timeout=self.timeout,
            )
        response_json(upload)
        self.wait_resource(f"/models/{model_uuid}", "model")
        return model_uuid

    def upload_calibration(self, path: Path, name: str) -> str:
        with path.open("rb") as zip_file:
            upload = self.session.post(
                f"{self.backend}/datasets",
                files={"dataset_file": (path.name, zip_file, "application/zip")},
                data={"dataset_name": name, "dataset_type": "calibration"},
                timeout=self.timeout,
            )
        payload = response_json(upload)
        dataset_uuid = str(payload["data"]["dataset"]["uuid"])
        self.wait_resource(f"/datasets/{dataset_uuid}", "dataset")
        return dataset_uuid

    def wait_resource(self, path: str, resource_type: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            payload = self._get(path)
            resource = payload["data"][resource_type]
            status = resource["status"]
            if status == "ready":
                return resource
            if status in {"error", "failed"}:
                raise EiqError(f"{resource_type} failed: {resource.get('status_description', resource)}")
            time.sleep(self.poll_seconds)
        raise EiqError(f"timed out waiting for {resource_type} at {path}")

    def run_optimization(self, model_uuid: str, passes: list[dict[str, Any]]) -> str:
        response = self.session.post(
            f"{self.backend}/optimizations/run",
            json={"model_uuid": model_uuid, "passes": passes},
            timeout=self.timeout,
        )
        payload = response_json(response)
        return str(payload["data"]["optimization"]["uuid"])

    def wait_optimization(self, optimization_uuid: str) -> str:
        deadline = time.monotonic() + self.timeout
        path = f"/optimizations/{optimization_uuid}"
        while time.monotonic() < deadline:
            payload = self._get(path)
            optimization = payload["data"]["optimization"]
            status = optimization["status"]
            if status == "success":
                artifacts = optimization.get("artifacts", [])
                if not artifacts:
                    raise EiqError("optimization succeeded but returned no artifacts")
                return str(artifacts[0]["artifact_id"])
            if status in {"error", "failed"}:
                raise EiqError(f"optimization failed: {optimization}")
            time.sleep(self.poll_seconds)
        raise EiqError(f"timed out waiting for optimization {optimization_uuid}")

    def download_artifact(self, optimization_uuid: str, artifact_id: str, destination: Path) -> Path:
        response = self.session.get(
            f"{self.backend}/optimizations/{optimization_uuid}/resources/{artifact_id}",
            timeout=self.timeout,
        )
        if not response.ok:
            response_json(response)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        if not destination.stat().st_size:
            raise EiqError(f"downloaded empty artifact to {destination}")
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="eIQ AI Toolkit: ONNX2Quant then TFLiteConversion")
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--dataset-name", default="depth-anything-metric-small-calibration")
    parser.add_argument("--tflite-output", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--quantized-onnx-output", type=Path, default=Path("artifacts/depth_anything_metric_small_int8.onnx"))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=1800.0, help="per backend request or polling phase")
    parser.add_argument("--execute", action="store_true", help="perform uploads and conversion jobs")
    args = parser.parse_args()
    if not args.onnx.is_file():
        parser.error(f"ONNX model not found: {args.onnx}")
    sample_count = validate_calibration_zip(args.calibration)
    if not args.execute:
        print(
            "dry run OK\n"
            f"backend: {args.backend}\nonnx: {args.onnx}\ncalibration samples: {sample_count}\n"
            "pass --execute to upload the model/dataset and start ONNX2Quant + TFLiteConversion"
        )
        return

    client = EiqClient(args.backend, timeout=args.timeout, poll_seconds=args.poll_seconds)
    print("checking eIQ backend passes…")
    client.check_backend()
    print("uploading FP32 ONNX…")
    source_model_uuid = client.upload_model(args.onnx)
    print("uploading calibration ZIP…")
    dataset_uuid = client.upload_calibration(args.calibration, args.dataset_name)
    print("running ONNX2Quant…")
    quant_job = client.run_optimization(
        source_model_uuid,
        [{"type": "ONNX2Quant", "config": {"allow_opset_10_and_lower": "false", "dataset_uuid": dataset_uuid}}],
    )
    quant_artifact = client.wait_optimization(quant_job)
    quantized_onnx = client.download_artifact(quant_job, quant_artifact, args.quantized_onnx_output)
    print(f"quantized ONNX: {quantized_onnx}")
    print("uploading quantized ONNX…")
    quantized_model_uuid = client.upload_model(quantized_onnx)
    print("running TFLiteConversion…")
    tflite_job = client.run_optimization(quantized_model_uuid, [{"type": "TFLiteConversion", "config": {}}])
    tflite_artifact = client.wait_optimization(tflite_job)
    tflite = client.download_artifact(tflite_job, tflite_artifact, args.tflite_output)
    print(f"TFLite conversion complete: {tflite}")


if __name__ == "__main__":
    main()
