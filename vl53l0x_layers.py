"""VL53L0X ranging and persistent two-layer calibration for the drawer probe."""
from __future__ import annotations

import fcntl
import json
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from adafruit_vl53l0x import VL53L0X


_I2C_SLAVE = 0x0703


class LinuxI2C:
    """Minimal CircuitPython-compatible I2C wrapper over Linux i2c-dev."""

    def __init__(self, device: str) -> None:
        self.fd = os.open(device, os.O_RDWR)
        self._lock = threading.Lock()

    def try_lock(self) -> bool:
        return self._lock.acquire(blocking=False)

    def unlock(self) -> None:
        self._lock.release()

    def _select(self, address: int) -> None:
        fcntl.ioctl(self.fd, _I2C_SLAVE, address)

    def writeto(self, address: int, buffer, *, start: int = 0, end: Optional[int] = None) -> None:
        self._select(address)
        end = len(buffer) if end is None else end
        data = bytes(memoryview(buffer)[start:end])
        if data:
            os.write(self.fd, data)

    def readfrom_into(self, address: int, buffer, *, start: int = 0, end: Optional[int] = None) -> None:
        self._select(address)
        end = len(buffer) if end is None else end
        data = os.read(self.fd, end - start)
        if len(data) != end - start:
            raise OSError("short I2C read")
        buffer[start:end] = data

    def writeto_then_readfrom(
        self,
        address: int,
        out_buffer,
        in_buffer,
        *,
        out_start: int = 0,
        out_end: Optional[int] = None,
        in_start: int = 0,
        in_end: Optional[int] = None,
    ) -> None:
        self.writeto(address, out_buffer, start=out_start, end=out_end)
        self.readfrom_into(address, in_buffer, start=in_start, end=in_end)

    def close(self) -> None:
        os.close(self.fd)


class DistanceSensor:
    def __init__(self, device: str = "/dev/i2c-0", address: int = 0x29) -> None:
        self.bus = LinuxI2C(device)
        try:
            self.sensor = VL53L0X(self.bus, address=address, io_timeout_s=1.0)
            self.sensor.start_continuous()
        except Exception:
            self.bus.close()
            raise

    def read_mm(self) -> int:
        value = int(self.sensor.range)
        if not 30 <= value <= 2000:
            raise RuntimeError(f"invalid VL53L0X range: {value} mm")
        return value

    def median_mm(self, samples: int = 15, delay_s: float = 0.04) -> tuple[float, float]:
        values = []
        for _ in range(samples * 5):
            try:
                values.append(self.read_mm())
            except (OSError, RuntimeError):
                pass
            if len(values) == samples:
                break
            time.sleep(delay_s)
        if len(values) < samples:
            raise RuntimeError("VL53L0X has no valid target between 30 and 2000 mm")
        median = float(statistics.median(values))
        mad = float(statistics.median(abs(value - median) for value in values))
        return median, mad

    def close(self) -> None:
        try:
            self.sensor.stop_continuous()
        finally:
            self.bus.close()


@dataclass(frozen=True)
class LayerProfiles:
    distances_mm: dict[int, float]
    tolerance_mm: float

    @classmethod
    def load(cls, path: Path) -> "LayerProfiles":
        data = json.loads(path.read_text(encoding="utf-8"))
        distances = {int(layer): float(value) for layer, value in data["distances_mm"].items()}
        if set(distances) != {1, 2}:
            raise ValueError("VL53 calibration must contain layers 1 and 2")
        return cls(distances, float(data["tolerance_mm"]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "distances_mm": {str(layer): value for layer, value in self.distances_mm.items()},
                    "tolerance_mm": self.tolerance_mm,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def match(self, distance_mm: float) -> Optional[int]:
        ranked = sorted((abs(distance_mm - baseline), layer) for layer, baseline in self.distances_mm.items())
        if ranked[0][0] > self.tolerance_mm or ranked[0][0] == ranked[1][0]:
            return None
        return ranked[0][1]


def make_profiles(readings: dict[int, float], noise: list[float]) -> LayerProfiles:
    if set(readings) != {1, 2} or len(noise) != 2:
        raise ValueError("layer 1 and 2 readings are required")
    separation = abs(readings[1] - readings[2])
    if separation < 30:
        raise RuntimeError(f"layer distances are only {separation:.0f} mm apart; reposition the sensor")
    tolerance = min(80.0, separation * 0.45)
    if tolerance < max(15.0, max(noise) * 6.0):
        raise RuntimeError("VL53L0X readings are too noisy to distinguish the two layers")
    return LayerProfiles(readings, tolerance)


def calibrate_two_layers(sensor: DistanceSensor, samples: int = 15) -> LayerProfiles:
    readings: dict[int, float] = {}
    noise: list[float] = []
    for layer in (1, 2):
        input(f"Open only drawer layer {layer}, keep it still, then press Enter...")
        median, mad = sensor.median_mm(samples)
        readings[layer] = median
        noise.append(mad)
        print(f"layer {layer}: {median:.0f} mm (MAD {mad:.1f} mm)")
    return make_profiles(readings, noise)
