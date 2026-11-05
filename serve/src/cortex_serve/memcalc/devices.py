"""Device table loader."""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class Device(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["gpu", "cpu", "unified"]
    memory_gib: float
    bandwidth_gbs: float | None
    bf16_tflops: float | None
    fp8_tflops: float | None
    source_url: str

    @property
    def memory_bytes(self) -> int:
        return int(self.memory_gib * 2**30)


@cache
def load_devices() -> dict[str, Device]:
    raw = yaml.safe_load(files("cortex_serve.memcalc").joinpath("devices.yaml").read_text())
    return {name: Device(name=name, **spec) for name, spec in raw.items()}


def get_device(name: str) -> Device:
    devices = load_devices()
    lookup = {k.lower(): v for k, v in devices.items()}
    try:
        return lookup[name.lower()]
    except KeyError:
        raise ValueError(f"unknown device {name!r}; choose from {sorted(devices)}") from None
