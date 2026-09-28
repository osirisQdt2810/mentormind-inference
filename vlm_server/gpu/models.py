"""What the server needs to know about a GPU, whatever the vendor."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: GPU vendor stack the server runs on.
Platform = Literal["cuda", "rocm"]


class GpuStatus(BaseModel):
    """One GPU as reported by the vendor tool (nvidia-smi / rocm-smi / amd-smi)."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    used_mib: int = Field(ge=0)
    total_mib: int = Field(gt=0)
    utilization_pct: int = Field(default=0, ge=0, le=100)

    @property
    def free_gib(self) -> float:
        return (self.total_mib - self.used_mib) / 1024

    @property
    def total_gib(self) -> float:
        return self.total_mib / 1024
