"""DEV/TEST ONLY: an 'oracle' matte = ground-truth alpha of the synthetic plate, for exactly a job's
region/frames, produced through the same geometry as load_job (so it is what a perfect model would return).

The synthetic plate must have an `i2roto_synth.json` next to it (written by tests/make_plate.py).
"""
import os

import numpy as np

from .io_utils import I2RotoError
from .synth import Synth
from . import merge as M

_cache = {}


def synth_for(run):
    spec = run.plate_spec
    d = spec if os.path.isdir(spec) else os.path.dirname(spec)
    path = os.path.join(d, "i2roto_synth.json")
    if not os.path.exists(path):
        raise I2RotoError(f"Stub matte: no i2roto_synth.json next to the plate ({path}). "
                          f"The stub only works on the synthetic test plate.")
    if path not in _cache:
        _cache[path] = Synth.load(path)
    s = _cache[path]
    if (s.W, s.H) != (run.width, run.height):
        raise I2RotoError("Stub matte: synth params do not match the plate size.")
    return s


def oracle_frame(run, job, synth, t):
    """Ground-truth alpha for source frame t in the job window (float32 win_h x win_w)."""
    if job["stage"] == "detail" and run.d >= 1.0:
        a = synth.alpha_region(t, job["x"], job["y"], job["w"], job["h"])
    else:
        full = synth.alpha(t)
        if job["stage"] == "proxy":
            a = M.resize(full, job["w"], job["h"], "area")
        else:
            a = M.resize(full, run.work_w, run.work_h, "area")[job["y"]:job["y"] + job["h"], job["x"]:job["x"] + job["w"]]
    return np.pad(a, ((0, job["win_h"] - job["h"]), (0, job["win_w"] - job["w"])), mode="edge")


def oracle_job(run, job):
    synth = synth_for(run)
    out = np.empty((job["length"], job["win_h"], job["win_w"]), dtype=np.float32)
    last_t, last = None, None
    for pos, f in enumerate(job["frames"]):
        t = run.start + f
        if t != last_t:
            last = oracle_frame(run, job, synth, t)
            last_t = t
        out[pos] = last
    return out
