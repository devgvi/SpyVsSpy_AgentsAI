"""M5E.1.1 spec sections 28-29 / M5E.1.2 spec sections 13, 19-20: cheap,
best-effort resource telemetry via `ps`/`vm_stat`/`sysctl`/`memory_pressure`
(no new dependency - `psutil` isn't installed in this environment and
adding it purely for a diagnostic milestone wasn't judged worth it). GPU/
Metal telemetry (M5E.1.1 section 29 / M5E.1.2 section 21) is explicitly
optional and not collected here - this platform exposes no cheap,
scriptable equivalent of `nvidia-smi` that doesn't require `powermetrics`
(privileged, explicitly disallowed by M5E.1.2 section 21); documented as a
limitation rather than guessed at. Every command here is a plain, safe,
non-privileged macOS system tool - no `sudo` anywhere (spec section 20).
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from typing import Any


def find_ollama_pid() -> int | None:
    try:
        out = subprocess.run(["pgrep", "-f", "ollama serve"], capture_output=True, text=True, timeout=5.0)
        pids = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
        return pids[0] if pids else None
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def sample_process(pid: int) -> dict[str, Any] | None:
    """RSS (KB->bytes), %CPU, thread count via `ps`/`lsof` - None for any
    field `ps` doesn't report, never a fabricated value."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=,pcpu=,nlwp=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        line = out.stdout.strip()
        if not line:
            return None
        parts = line.split()
        rss_kb = int(parts[0]) if len(parts) > 0 else None
        pcpu = float(parts[1]) if len(parts) > 1 else None
        # macOS `ps` has no `nlwp` (that's Linux-only) - thread count is
        # collected only where the platform's `ps` actually reports it.
        nlwp = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
        fd_count = _open_fd_count(pid)
        return {"rss_bytes": (rss_kb * 1024) if rss_kb is not None else None, "cpu_percent": pcpu, "thread_count": nlwp, "open_fd_count": fd_count}
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None


def _open_fd_count(pid: int) -> int | None:
    try:
        out = subprocess.run(["lsof", "-p", str(pid)], capture_output=True, text=True, timeout=8.0)
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        return max(len(lines) - 1, 0) if lines else None  # first line is the header
    except (subprocess.SubprocessError, OSError):
        return None


def sample_resources(label: str, ollama_pid: int | None, include_self: bool = True) -> dict[str, Any]:
    import os

    return {
        "label": label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ollama": sample_process(ollama_pid) if ollama_pid is not None else None,
        "diagnostic_process": sample_process(os.getpid()) if include_self else None,
    }


# -- M5E.1.2: macOS system memory metrics (spec sections 13, 19-20) --------

_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
_VM_STAT_LINE_RE = re.compile(r"^(.+?):\s+(\d+)\.?$")


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout
    except (subprocess.SubprocessError, OSError):
        return None


def sample_vm_stat() -> dict[str, Any] | None:
    """Parses `vm_stat` - page size plus every `Pages <label>: N.` line we
    care about. Compressed memory = "Pages occupied by compressor" *
    page_size (the actual resident cost of compression, distinct from
    "Pages stored in compressor" which counts logical, pre-compression
    pages)."""
    text = _run(["vm_stat"])
    if not text:
        return None
    size_match = _PAGE_SIZE_RE.search(text)
    if not size_match:
        return None
    page_size = int(size_match.group(1))
    pages: dict[str, int] = {}
    for line in text.splitlines()[1:]:
        m = _VM_STAT_LINE_RE.match(line.strip())
        if m:
            pages[m.group(1)] = int(m.group(2))

    def bytes_for(label: str) -> int | None:
        return pages[label] * page_size if label in pages else None

    return {
        "page_size_bytes": page_size,
        "free_bytes": bytes_for("Pages free"),
        "active_bytes": bytes_for("Pages active"),
        "inactive_bytes": bytes_for("Pages inactive"),
        "wired_bytes": bytes_for("Pages wired down"),
        "compressed_bytes": bytes_for("Pages occupied by compressor"),
        "swapins": pages.get("Swapins"),
        "swapouts": pages.get("Swapouts"),
    }


def sample_swap_usage() -> dict[str, Any] | None:
    """Parses `sysctl vm.swapusage`: "vm.swapusage: total = 3072.00M used
    = 1404.94M free = 1667.06M (encrypted)"."""
    text = _run(["sysctl", "vm.swapusage"])
    if not text:
        return None
    numbers = re.findall(r"=\s*([\d.]+)M", text)
    if len(numbers) < 3:
        return None
    return {
        "total_mb": float(numbers[0]),
        "used_mb": float(numbers[1]),
        "free_mb": float(numbers[2]),
    }


def sample_memory_pressure() -> dict[str, Any] | None:
    """`memory_pressure` without `-l`/`-p`/`-w` only *reports*, allocating
    nothing (safe, no side effects) - its one directly reported number
    without sudo is "System-wide memory free percentage." There is no
    non-privileged way to read the kernel's own normal/warn/critical
    dispatch-source level (that needs elevated access, explicitly out of
    scope per spec section 20) - `pressure_state` below is this module's
    own descriptive label derived from that free percentage, not the
    kernel's classification, and is documented as such everywhere it's
    used."""
    text = _run(["memory_pressure"], timeout=10.0)
    if not text:
        return None
    m = re.search(r"free percentage:\s*(\d+)%", text)
    if not m:
        return None
    free_percent = int(m.group(1))
    if free_percent >= 50:
        pressure_state = "healthy (derived, not kernel-reported)"
    elif free_percent >= 20:
        pressure_state = "elevated (derived, not kernel-reported)"
    else:
        pressure_state = "high (derived, not kernel-reported)"
    return {"free_percent": free_percent, "pressure_state": pressure_state}


def sample_physical_memory_bytes() -> int | None:
    text = _run(["sysctl", "hw.memsize"])
    if not text:
        return None
    m = re.search(r"hw\.memsize:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def sample_system_memory() -> dict[str, Any]:
    """Everything spec section 19's "system:" block asks for, from the
    three safe, non-privileged sources above. Any field whose source
    command failed is `None`, never fabricated."""
    vm = sample_vm_stat()
    swap = sample_swap_usage()
    pressure = sample_memory_pressure()
    return {
        "physical_memory_bytes": sample_physical_memory_bytes(),
        "compressed_bytes": vm.get("compressed_bytes") if vm else None,
        "free_bytes": vm.get("free_bytes") if vm else None,
        "wired_bytes": vm.get("wired_bytes") if vm else None,
        "swap_used_mb": swap.get("used_mb") if swap else None,
        "swap_total_mb": swap.get("total_mb") if swap else None,
        "memory_free_percent": pressure.get("free_percent") if pressure else None,
        "memory_pressure_state": pressure.get("pressure_state") if pressure else None,
    }


def sample_model_residency(base_url: str = "http://localhost:11434", model_name: str = "gemma4:latest") -> dict[str, Any]:
    """`/api/ps` - whether the model is currently loaded, and whatever
    residency/size/expiry fields Ollama actually reports for it (spec
    section 36: "do not assume undocumented fields exist")."""
    from src.evaluation.ollama_health import _get_json

    payload = _get_json(f"{base_url.rstrip('/')}/api/ps")
    if payload is None:
        return {"resident": None, "details": None}
    models = payload.get("models") if isinstance(payload, dict) else None
    entry = next((m for m in (models or []) if m.get("model") == model_name or m.get("name") == model_name), None)
    if entry is None:
        return {"resident": False, "details": None}
    return {
        "resident": True,
        "details": {
            "size_bytes": entry.get("size"),
            "size_vram_bytes": entry.get("size_vram"),
            "context_length": entry.get("context_length"),
            "expires_at": entry.get("expires_at"),
        },
    }


def sample_full_resource_state(
    label: str, ollama_pid: int | None, base_url: str = "http://localhost:11434", model_name: str = "gemma4:latest"
) -> dict[str, Any]:
    """The complete per-sample record spec section 19 asks for: system
    memory, Ollama process, benchmark process, and model residency
    together, one timestamp shared across all four."""
    base = sample_resources(label, ollama_pid)
    base["system"] = sample_system_memory()
    base["model"] = sample_model_residency(base_url, model_name)
    return base
