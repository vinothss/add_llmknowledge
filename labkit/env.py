"""Runtime detection and per-tier defaults.

Every notebook calls `rt = labkit.env.setup()` first. It detects whether we
are on Colab Free (T4), Colab Pro (L4 / A100 / H100), Apple Silicon, or CPU,
and picks model sizes so the same notebook runs everywhere.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

# Models per tier. Free-tier defaults are ungated (no Hugging Face token needed).
MODELS = {
    "pro":  {"generator": "Qwen/Qwen2.5-7B-Instruct",   "student": "Qwen/Qwen2.5-1.5B-Instruct"},
    "free": {"generator": "Qwen/Qwen2.5-1.5B-Instruct", "student": "Qwen/Qwen2.5-0.5B-Instruct"},
    "mps":  {"generator": "Qwen/Qwen2.5-1.5B-Instruct", "student": "Qwen/Qwen2.5-0.5B-Instruct"},
    "cpu":  {"generator": "Qwen/Qwen2.5-0.5B-Instruct", "student": "Qwen/Qwen2.5-0.5B-Instruct"},
}


@dataclass
class Runtime:
    tier: str                 # "pro" | "free" | "mps" | "cpu"
    device: str               # "cuda" | "mps" | "cpu"
    gpu_name: str = "none"
    vram_gb: float = 0.0
    supports_bf16: bool = False
    supports_fp8: bool = False
    workdir: Path = field(default_factory=Path)
    repo_dir: Path = field(default_factory=Path)

    @property
    def is_pro(self) -> bool:
        return self.tier == "pro"

    def model(self, role: str) -> str:
        return os.environ.get(f"LAB_{role.upper()}_MODEL", MODELS[self.tier][role])


def _detect() -> dict:
    try:
        import torch
    except ImportError:
        return {"tier": "cpu", "device": "cpu"}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1e9
        major, minor = torch.cuda.get_device_capability(0)
        return {
            "tier": "pro" if vram >= 22 else "free",
            "device": "cuda",
            "gpu_name": props.name,
            "vram_gb": round(vram, 1),
            "supports_bf16": major >= 8,
            # FP8 needs Ada (8.9) or Hopper (9.0+): L4, L40S, H100.
            "supports_fp8": (major, minor) >= (8, 9),
        }
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"tier": "mps", "device": "mps", "gpu_name": "Apple Silicon"}
    return {"tier": "cpu", "device": "cpu"}


def _find_repo() -> Path:
    here = Path.cwd()
    for p in [here, *here.parents]:
        if (p / "labkit").is_dir() and (p / "data").is_dir():
            return p
    return here


def setup(use_drive: bool = False, force_tier: str | None = None) -> Runtime:
    """Detect hardware, choose a working directory, and print a banner."""
    info = _detect()
    if force_tier or os.environ.get("LAB_TIER"):
        info["tier"] = force_tier or os.environ["LAB_TIER"]

    if IN_COLAB and use_drive:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        workdir = Path("/content/drive/MyDrive/llm-knowledge-workshop")
    elif IN_COLAB:
        workdir = Path("/content/work")
    else:
        workdir = _find_repo() / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    rt = Runtime(**info, workdir=workdir, repo_dir=_find_repo())
    from . import progress
    progress.bind(workdir)
    label = {
        "pro": "Colab Pro track  (bonus exercises enabled)",
        "free": "Colab Free track",
        "mps": "Local Apple Silicon (small models)",
        "cpu": "CPU only (tiny models - expect slow generation)",
    }[rt.tier]
    print(f"Runtime : {label}")
    print(f"GPU     : {rt.gpu_name}  {rt.vram_gb or ''} {'GB' if rt.vram_gb else ''}")
    print(f"bf16/fp8: {rt.supports_bf16} / {rt.supports_fp8}")
    print(f"Workdir : {rt.workdir}{'  (saved to Google Drive)' if use_drive and IN_COLAB else ''}")
    if rt.tier in ("cpu", "mps") and IN_COLAB:
        print("\nTip: Runtime > Change runtime type > T4 GPU for a much faster lab.")
    return rt
