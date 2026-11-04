"""
app/api/routes/health.py
────────────────────────
System health and readiness check.
"""
from __future__ import annotations

import psutil
from datetime import datetime, timezone
from fastapi import APIRouter
from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Return health status, version, and basic hardware telemetry."""
    settings = get_settings()
    mem = psutil.virtual_memory()

    cuda_available = False
    cuda_device = "N/A"
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "hardware": {
            "ram_total_gb": round(mem.total / 1e9, 2),
            "ram_available_gb": round(mem.available / 1e9, 2),
            "ram_percent": mem.percent,
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "cuda_available": cuda_available,
            "cuda_device": cuda_device,
        },
    }

