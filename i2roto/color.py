"""Plate colour handling: turn whatever the plate is into a normal-looking display-space picture.

This only affects what the model sees (and the optional RGBA/comp previews). It never touches the output matte.
"""
import numpy as np

COLOR_AS_IS = "rec709 / sRGB (use as is)"
COLOR_LINEAR = "linear (EXR) -> sRGB"
COLOR_LOG = "log -> auto normalize"
COLOR_MODES = [COLOR_AS_IS, COLOR_LINEAR, COLOR_LOG]

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def srgb_oetf(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def estimate_log_params(frames):
    """frames: iterable of float32 HxWx3 (any size, typically downscaled samples).

    Returns dict(lo, hi, gamma) for the log -> auto normalize mode.
    """
    lumas = []
    for f in frames:
        y = (f[:, :, :3] * LUMA).sum(axis=2)
        lumas.append(y.reshape(-1)[:: max(1, y.size // 200000)])
    y = np.concatenate(lumas).astype(np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return {"lo": 0.0, "hi": 1.0, "gamma": 1.0}
    lo, hi = np.percentile(y, [0.5, 99.5])
    if hi - lo < 1e-4:
        hi = lo + 1e-4
    ys = np.clip((y - lo) / (hi - lo), 0.0, 1.0)
    med = float(np.median(ys))
    med = min(max(med, 0.02), 0.98)
    gamma = float(np.log(0.45) / np.log(med))
    gamma = min(max(gamma, 0.6), 1.6)  # "mild"
    return {"lo": float(lo), "hi": float(hi), "gamma": gamma}


def to_display(rgb, mode, log_params=None):
    """rgb: float32 HxWx3 as read from the plate. Returns float32 HxWx3 in 0..1."""
    if mode == COLOR_LINEAR:
        return srgb_oetf(np.maximum(rgb, 0.0))
    if mode == COLOR_LOG:
        p = log_params or {"lo": 0.0, "hi": 1.0, "gamma": 1.0}
        x = np.clip((rgb - p["lo"]) / (p["hi"] - p["lo"]), 0.0, 1.0)
        x = np.power(x, p["gamma"], dtype=np.float32)
        y = (x * LUMA).sum(axis=2, keepdims=True)
        x = y + (x - y) * 1.25
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
