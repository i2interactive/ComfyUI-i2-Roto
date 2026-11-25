"""Core of the five nodes, as plain functions on numpy arrays (no ComfyUI, no torch).

The run state (RunState) is created by setup(), passed along the graph as I2ROTO_RUN, and keeps every
intermediate result on disk in its cache dir so RAM stays bounded.
"""
import json
import logging
import math
import os
import shutil
import time
import uuid

import numpy as np

from . import plan as P
from . import merge as M
from .color import COLOR_MODES, COLOR_LOG, to_display, estimate_log_params
from .io_utils import I2RotoError, clean_path, read_gray, write_png16_gray, write_matte, write_rgba16, get_cv2
from .plate import open_plate

log = logging.getLogger("i2roto")

OUTPUT_FORMATS = {"PNG 16-bit": ("png16", "png"), "EXR half float": ("exr", "exr"), "TIFF 16-bit": ("tiff16", "tif")}
OUTPUT_FORMAT_NAMES = list(OUTPUT_FORMATS)

DEFAULTS = {
    "plate_path": "", "start_frame": 0, "frame_count": 0, "plate_color": COLOR_MODES[0],
    "window": P.DEFAULT_WINDOW, "chunk_frames": 49, "detail": P.DETAIL_NATIVE, "tile_overlap": 256,
    "band_px": 48, "tile_trust": 1.0, "pin_chunks": True, "black_clip": 0.005, "white_clip": 0.995,
    "invert": False, "edge_shift_px": 0.0, "guided_upscale": True, "garbage_matte_path": "",
    "output_folder": "", "output_format": "PNG 16-bit", "write_rgba": False, "start_number": 1001,
    "keep_cache": False,
}


class RunState:
    """Everything a run needs. Small; frames live on disk. Plate handles are opened lazily."""

    def __init__(self):
        self._plate = None
        self._garbage = None

    def __repr__(self):
        return f"<i2roto run {getattr(self, 'run_id', '?')}>"

    # Not deep-copied / not serialised by anything in ComfyUI, but be safe.
    def __getstate__(self):
        d = dict(self.__dict__)
        d["_plate"] = None
        d["_garbage"] = None
        return d

    def plate(self):
        if self._plate is None:
            self._plate = open_plate(self.plate_spec)
        return self._plate

    def garbage(self):
        if self._garbage is None and self.garbage_spec:
            self._garbage = open_plate(self.garbage_spec)
        return self._garbage

    def close(self):
        for p in (self._plate, self._garbage):
            if p is not None:
                try:
                    p.close()
                except Exception:
                    pass
        self._plate = None
        self._garbage = None

    def display(self, raw):
        return to_display(raw, self.cfg["plate_color"], self.log_params)

    def save_json(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        with open(os.path.join(self.cache_dir, "run.json"), "w") as f:
            json.dump(d, f, indent=1, default=str)


# ============================================================================= helpers

def _pad_edge(img, ph, pw):
    """Pad HxW(xC) on the right/bottom with edge replication."""
    if ph == 0 and pw == 0:
        return img
    pad = ((0, ph), (0, pw)) + (((0, 0),) if img.ndim == 3 else ())
    return np.pad(img, pad, mode="edge")


def _fit_width(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    nh = max(1, int(round(h * max_w / float(w))))
    return M.resize(img, max_w, nh, "area")


def resolve_output_folder(cfg, base, output_dir):
    p = clean_path(cfg.get("output_folder", ""))
    if not p:
        return os.path.join(output_dir, "i2roto", f"{base}_matte")
    if not os.path.isabs(p):
        return os.path.normpath(os.path.join(output_dir, p))
    return p


# ============================================================================= 1. Setup

def setup(cfg_in, temp_dir, output_dir):
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in cfg_in.items() if k in DEFAULTS})
    if cfg["plate_color"] not in COLOR_MODES:
        raise I2RotoError(f"Unknown plate_color: {cfg['plate_color']}")
    if cfg["window"] not in P.WINDOWS:
        raise I2RotoError(f"Unknown window: {cfg['window']}")
    if cfg["detail"] not in P.DETAIL_MODES:
        raise I2RotoError(f"Unknown detail mode: {cfg['detail']}")
    if cfg["output_format"] not in OUTPUT_FORMATS:
        raise I2RotoError(f"Unknown output_format: {cfg['output_format']}")

    run = RunState()
    run.cfg = cfg
    run.plate_spec = clean_path(cfg["plate_path"])
    plate = open_plate(run.plate_spec)
    run._plate = plate
    run.kind = plate.kind
    run.base_name = plate.base_name
    run.width, run.height = int(plate.width), int(plate.height)
    run.fps = plate.fps
    total = int(plate.frame_count)
    start = int(cfg["start_frame"])
    if total <= 0:
        raise I2RotoError(f"The plate has zero frames: {run.plate_spec}")
    if start < 0 or start >= total:
        raise I2RotoError(f"start_frame {start} is outside the plate (it has {total} frames, 0..{total - 1}).")
    avail = total - start
    n = avail if int(cfg["frame_count"]) <= 0 else min(int(cfg["frame_count"]), avail)
    run.total_frames, run.start, run.n = total, start, n
    if plate.kind == "sequence" and plate.numbers is not None:
        run.out_numbers = [plate.numbers[start + i] for i in range(n)]
    else:
        run.out_numbers = [int(cfg["start_number"]) + i for i in range(n)]

    run.window = P.WINDOWS[cfg["window"]]
    run.L = P.snap_chunk_frames(cfg["chunk_frames"])
    run.geom = P.proxy_geometry(run.width, run.height, *run.window)
    run.chunks = P.chunk_plan(n, run.L)
    run.s = run.geom["s"]
    run.d = P.working_scale(cfg["detail"], run.s)
    run.work_w = int(round(run.width * run.d)) if run.d < 1 else run.width
    run.work_h = int(round(run.height * run.d)) if run.d < 1 else run.height

    # Colour: per-shot log normalisation from ~12 frames.
    run.log_params = None
    if cfg["plate_color"] == COLOR_LOG:
        idx = sorted(set(int(round(x)) for x in np.linspace(start, start + n - 1, min(12, n))))
        samples = [_fit_width(f, 512) for _, f in plate.iter_frames(idx)]
        run.log_params = estimate_log_params(samples)

    # Garbage matte: validate now (fail early).
    run.garbage_spec = clean_path(cfg["garbage_matte_path"])
    if run.garbage_spec:
        g = open_plate(run.garbage_spec)
        run.garbage_count = int(g.frame_count)
        run._garbage = g

    run.out_dir = resolve_output_folder(cfg, run.base_name, output_dir)
    run.fmt, run.out_ext = OUTPUT_FORMATS[cfg["output_format"]]
    try:
        os.makedirs(run.out_dir, exist_ok=True)
    except OSError as e:
        raise I2RotoError(f"Cannot create the output folder {run.out_dir}: {e}")
    run.rgba_dir = run.out_dir.rstrip("\\/") + "_rgba" if cfg["write_rgba"] else None

    run.run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run.cache_dir = os.path.join(temp_dir, "i2roto", run.run_id)
    os.makedirs(os.path.join(run.cache_dir, "proxy"), exist_ok=True)
    run.detail_tiles = []
    run.detail_jobs = []
    run.written = {"proxy": [], "detail": []}
    run.warnings = []
    run.t_start = time.time()
    run.save_json()

    first = _fit_width(run.display(plate.read(start)), 1024)
    blank = np.zeros((1, run.geom["win_h"], run.geom["win_w"], 3), dtype=np.float32)
    info = setup_info(run)
    return run, len(run.chunks), blank, first[None], info


def setup_info(run):
    g = run.geom
    lines = [
        f"Plate: {run.plate_spec}",
        f"  {run.kind}, {run.width}x{run.height}, {run.total_frames} frames"
        + (f", {run.fps:.3f} fps" if run.fps else "") + f"; using frames {run.start}..{run.start + run.n - 1} ({run.n})",
        f"  output numbers {run.out_numbers[0]}..{run.out_numbers[-1]}",
        f"Proxy: scale {g['s']:.4f} -> {g['pw']}x{g['ph']} padded to {g['win_w']}x{g['win_h']}",
        f"Chunks: {len(run.chunks)} x {run.chunks[0]['length']} frames (chunk_frames {run.L})",
        f"Detail: {run.cfg['detail']}; working resolution {run.work_w}x{run.work_h} (scale {run.d:.4f})",
        f"Output: {run.out_dir} ({run.cfg['output_format']})",
        f"Cache: {run.cache_dir}",
    ]
    if run.log_params:
        lp = run.log_params
        lines.append(f"Log normalize: lo {lp['lo']:.4f} hi {lp['hi']:.4f} gamma {lp['gamma']:.3f}")
    return "\n".join(lines)


# ============================================================================= 2. Load job

def get_job(run, stage, index):
    if stage == "proxy":
        if index < 0 or index >= len(run.chunks):
            raise I2RotoError(f"Proxy job {index} does not exist (there are {len(run.chunks)}).")
        ch = run.chunks[index]
        g = run.geom
        return {"stage": "proxy", "index": index, "chunk": ch["index"], "tile": None,
                "frames": ch["frames"], "own": ch["own"], "length": ch["length"],
                "w": g["pw"], "h": g["ph"], "win_w": g["win_w"], "win_h": g["win_h"],
                "is_first_chunk": (ch["index"] == 0) or not run.cfg["pin_chunks"], "run_id": run.run_id}
    if stage == "detail":
        if index < 0 or index >= len(run.detail_jobs):
            raise I2RotoError(f"Detail job {index} does not exist (there are {len(run.detail_jobs)}).")
        j = run.detail_jobs[index]
        t = run.detail_tiles[j["tile"]]
        ch = run.chunks[j["chunk"]]
        own = list(ch["own"])
        if ch["index"] > 0 and j["keep0"]:
            own = [(0, ch["start"])] + own
        return {"stage": "detail", "index": index, "chunk": ch["index"], "tile": t["index"],
                "frames": ch["frames"], "own": own, "length": ch["length"],
                "x": t["x"], "y": t["y"], "w": t["w"], "h": t["h"], "win_w": t["win_w"], "win_h": t["win_h"],
                "is_first_chunk": j["is_first_chunk"], "run_id": run.run_id}
    raise I2RotoError(f"Unknown stage {stage}")


def job_view(run, job, raw):
    """Map one raw native plate frame to the job's window (no colour conversion). Returns HxWx3 float32."""
    if job["stage"] == "proxy":
        img = M.resize(raw, job["w"], job["h"], "area")
    else:
        if run.d < 1:
            img = M.resize(raw, run.work_w, run.work_h, "area")
        else:
            img = raw
        img = img[job["y"]:job["y"] + job["h"], job["x"]:job["x"] + job["w"]]
    return _pad_edge(img, job["win_h"] - job["h"], job["win_w"] - job["w"])


def load_job(run, stage, index):
    job = get_job(run, stage, index)
    plate = run.plate()
    out = np.empty((job["length"], job["win_h"], job["win_w"], 3), dtype=np.float32)
    src = [run.start + f for f in job["frames"]]
    for pos, (_, raw) in enumerate(plate.iter_frames(src)):
        out[pos] = run.display(job_view(run, job, raw))
    return out, job


# ============================================================================= 3. Store job

def _cache_path(run, stage, tile, frame):
    if stage == "proxy":
        d = os.path.join(run.cache_dir, "proxy")
    else:
        d = os.path.join(run.cache_dir, "detail", f"t{tile:03d}")
    return d, os.path.join(d, f"f{frame:06d}.png")


def store_job(run, job, matte):
    """matte: float32 [L, H, W, C]. Writes owned frames; returns (next_pin [1,H,W,3], run)."""
    if matte.ndim == 3:
        matte = matte[..., None]
    L, H, W = job["length"], job["win_h"], job["win_w"]
    next_pin = matte[-1:, :, :, :3] if matte.shape[-1] >= 3 else np.repeat(matte[-1:, :, :, :1], 3, axis=3)
    alpha = matte[..., :3].max(axis=3) if matte.shape[-1] >= 3 else matte[..., 0]
    if alpha.shape[1] != H or alpha.shape[2] != W:
        msg = (f"{job['stage']} job {job['index']}: decoded matte is {alpha.shape[2]}x{alpha.shape[1]}, "
               f"expected {W}x{H}; resizing.")
        log.warning(msg)
        run.warnings.append(msg)
        alpha = np.stack([M.resize(a, W, H, "cubic") for a in alpha])
    if alpha.shape[0] != L:
        msg = f"{job['stage']} job {job['index']}: decoded {alpha.shape[0]} frames, expected {L}; trimming/padding."
        log.warning(msg)
        run.warnings.append(msg)
        if alpha.shape[0] > L:
            alpha = alpha[:L]
        else:
            alpha = np.concatenate([alpha, np.repeat(alpha[-1:], L - alpha.shape[0], axis=0)])
    d, _ = _cache_path(run, job["stage"], job["tile"], 0)
    os.makedirs(d, exist_ok=True)
    for pos, frame in job["own"]:
        _, path = _cache_path(run, job["stage"], job["tile"], frame)
        key = (job["tile"], frame) if job["stage"] == "detail" else frame
        if os.path.exists(path):
            msg = f"{job['stage']} frame {frame} tile {job['tile']} written twice (job {job['index']})"
            log.warning(msg)
            run.warnings.append(msg)
        write_png16_gray(path, np.clip(alpha[pos, :job["h"], :job["w"]], 0.0, 1.0), compression=1)
        run.written[job["stage"]].append(key)
    return next_pin.astype(np.float32), run


# ============================================================================= 4. Plan detail

def read_proxy(run, f):
    _, path = _cache_path(run, "proxy", None, f)
    if not os.path.exists(path):
        raise I2RotoError(f"Proxy matte for frame {f} is missing ({path}). Did the proxy loop run completely?")
    return read_gray(path)


def band_radius_proxy(run):
    return max(2.0, run.cfg["band_px"] * run.s)


def plan_detail(run):
    t0 = time.time()
    run.detail_tiles, run.detail_jobs = [], []
    missing = [f for f in range(run.n) if not os.path.exists(_cache_path(run, "proxy", None, f)[1])]
    if missing:
        raise I2RotoError(f"{len(missing)} proxy matte frames are missing (first: {missing[0]}). "
                          f"The proxy loop did not finish.")
    reason = None
    if run.cfg["detail"] == P.DETAIL_PROXY:
        reason = "detail = proxy only"
    elif run.s >= 1.0:
        reason = "the plate already fits the model window"
    r = band_radius_proxy(run)
    pw, ph = run.geom["pw"], run.geom["ph"]
    union = np.zeros((ph, pw), dtype=bool)
    bands = []
    band_px_total = 0
    for f in range(run.n):
        p = read_proxy(run, f)
        union |= p > 0.02
        b = M.edge_band(p, r)
        band_px_total += int(b.sum())
        bands.append(np.packbits(b))
    run.band_coverage = band_px_total / float(max(1, run.n * pw * ph))
    if reason is None and not union.any():
        reason = "the proxy matte is empty"
    if reason is not None:
        run.detail_reason = reason
        run.save_json()
        return run, 0, f"Detail pass: no jobs ({reason}). Band coverage {100 * run.band_coverage:.2f}% of the proxy."

    cv2 = get_cv2()
    ri = int(math.ceil(r))
    dil = cv2.dilate(union.astype(np.uint8), M._disk(ri)) > 0
    ys, xs = np.nonzero(dil)
    k = run.d / run.s  # proxy px -> working px
    ov = int(run.cfg["tile_overlap"])
    box = [xs.min() * k, ys.min() * k, (xs.max() + 1) * k, (ys.max() + 1) * k]
    # Expand by the blend ramp so the ramps sit outside the band.
    tw, th = min(run.window[0], run.work_w), min(run.window[1], run.work_h)
    rx, ry = max(0, min(ov, tw // 2 - 1)), max(0, min(ov, th // 2 - 1))
    box = [int(math.floor(box[0] - rx)), int(math.floor(box[1] - ry)),
           int(math.ceil(box[2] + rx)), int(math.ceil(box[3] + ry))]
    box = [max(0, box[0]), max(0, box[1]), min(run.work_w, box[2]), min(run.work_h, box[3])]
    tiles = P.tile_grid(box, run.work_w, run.work_h, run.window[0], run.window[1], ov)
    rects = []
    for t in tiles:
        x0 = int(math.floor(t["x"] / k)); y0 = int(math.floor(t["y"] / k))
        x1 = int(math.ceil((t["x"] + t["w"]) / k)); y1 = int(math.ceil((t["y"] + t["h"]) / k))
        rects.append((max(0, x0), max(0, y0), min(pw, x1), min(ph, y1)))
    touched = np.zeros((run.n, len(tiles)), dtype=bool)
    for f in range(run.n):
        b = np.unpackbits(bands[f], count=pw * ph).reshape(ph, pw).astype(bool)
        for ti, (x0, y0, x1, y1) in enumerate(rects):
            touched[f, ti] = b[y0:y1, x0:x1].any()
    sched = np.zeros((len(tiles), len(run.chunks)), dtype=bool)
    for c in run.chunks:
        real = sorted(set(c["frames"]))
        sched[:, c["index"]] = touched[real].any(axis=0)
    jobs = []
    for ti in range(len(tiles)):
        for ci in range(len(run.chunks)):
            if not sched[ti, ci]:
                continue
            prev = ci > 0 and sched[ti, ci - 1]
            jobs.append({"tile": ti, "chunk": ci, "is_first_chunk": (not prev) or not run.cfg["pin_chunks"],
                         "keep0": not prev})
    run.detail_box = box
    run.detail_tiles = tiles
    run.detail_jobs = jobs
    run.detail_reason = None
    run.save_json()
    used_tiles = len(set(j["tile"] for j in jobs))
    info = (f"Detail pass: {len(jobs)} jobs = {used_tiles} of {len(tiles)} tiles x up to {len(run.chunks)} chunks "
            f"at {run.work_w}x{run.work_h}; tile {tiles[0]['w']}x{tiles[0]['h']} (window "
            f"{tiles[0]['win_w']}x{tiles[0]['win_h']}), overlap {tiles[0]['ox']}/{tiles[0]['oy']} px; "
            f"band radius {r:.1f} px at proxy; band coverage {100 * run.band_coverage:.2f}%; "
            f"planned in {time.time() - t0:.1f}s")
    return run, len(jobs), info


# ============================================================================= 5. Assemble

def _tile_weights(run):
    return [M.tile_weight(t, run.work_w, run.work_h) for t in run.detail_tiles]


def assemble_frame(run, f, weights, plate_display=None, stats=None):
    """Compute the final native matte for output position f (before writing). Returns HxW float32."""
    cfg = run.cfg
    pw, ph = run.geom["pw"], run.geom["ph"]
    p_proxy = read_proxy(run, f)
    Ww, Wh = run.work_w, run.work_h
    A = np.clip(M.resize(p_proxy, Ww, Wh, "cubic"), 0.0, 1.0)
    r = band_radius_proxy(run)
    band = M.edge_band(p_proxy, r).astype(np.float32)
    r_work = r * run.d / run.s
    sigma = max(1.0, r_work / 3.0)
    band_soft_full = None
    if stats is not None:
        stats["band"] += float(band.mean())

    tiles_here = []
    for t in run.detail_tiles:
        _, path = _cache_path(run, "detail", t["index"], f)
        if os.path.exists(path):
            tiles_here.append((t, path))
    if tiles_here:
        x0 = min(t["x"] for t, _ in tiles_here); y0 = min(t["y"] for t, _ in tiles_here)
        x1 = max(t["x"] + t["w"] for t, _ in tiles_here); y1 = max(t["y"] + t["h"] for t, _ in tiles_here)
        num = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        den = np.zeros_like(num)
        for t, path in tiles_here:
            a = read_gray(path)
            if a.shape != (t["h"], t["w"]):
                a = M.resize(a, t["w"], t["h"], "cubic")
            w = weights[t["index"]]
            sl = (slice(t["y"] - y0, t["y"] - y0 + t["h"]), slice(t["x"] - x0, t["x"] - x0 + t["w"]))
            num[sl] += w * a
            den[sl] += w
        T = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
        cov = np.clip(den, 0.0, 1.0)
        # Soft band at working res, only over the tiles' area (+ blur margin).
        m = int(math.ceil(3 * sigma)) + 2
        bx0, by0 = max(0, x0 - m), max(0, y0 - m)
        bx1, by1 = min(Ww, x1 + m), min(Wh, y1 + m)
        k = run.s / run.d  # working px -> proxy px
        # Resample the proxy band onto the working-res crop (pixel-centre aligned affine).
        cv2 = get_cv2()
        Mx = np.array([[k, 0, (bx0 + 0.5) * k - 0.5], [0, k, (by0 + 0.5) * k - 0.5]], dtype=np.float64)
        bcrop = cv2.warpAffine(band, Mx, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)
        bcrop = M.gaussian(bcrop, sigma)
        bs = bcrop[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
        Pc = A[y0:y1, x0:x1]
        A[y0:y1, x0:x1] = np.clip(Pc + bs * cov * float(cfg["tile_trust"]) * (T - Pc), 0.0, 1.0)
        if stats is not None:
            stats["tile_frames"] += len(tiles_here)

    if Ww != run.width or Wh != run.height:
        An = np.clip(M.resize(A, run.width, run.height, "cubic"), 0.0, 1.0)
        if cfg["guided_upscale"] and plate_display is not None and band.any():
            band_soft_full = M.gaussian(M.resize(band, Ww, Wh, "linear"), sigma)
            I_work = M.resize(plate_display, Ww, Wh, "area")
            ma, mb = M.guided_coeffs(I_work, A, r=8, eps=1e-4)
            bu = M.resize(band_soft_full, run.width, run.height, "linear")
            ys, xs = np.nonzero(bu > 1e-3)
            if ys.size:
                ry0, ry1, rx0, rx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                mau = M.resize(ma, run.width, run.height, "linear")[ry0:ry1, rx0:rx1]
                mbu = M.resize(mb, run.width, run.height, "linear")[ry0:ry1, rx0:rx1]
                q = np.clip((mau * plate_display[ry0:ry1, rx0:rx1]).sum(axis=2) + mbu, 0.0, 1.0)
                sub = An[ry0:ry1, rx0:rx1]
                bsub = bu[ry0:ry1, rx0:rx1]
                An[ry0:ry1, rx0:rx1] = sub + bsub * (q - sub)
        A = An

    A = M.edge_shift(A, float(cfg["edge_shift_px"]))
    return np.clip(A, 0.0, 1.0).astype(np.float32)


def _garbage_frame(run, f, cursor_state):
    g = run.garbage()
    if g is None:
        return None
    if g.frame_count == 1:
        gi = 0
    elif g.frame_count >= run.total_frames:
        gi = run.start + f
    else:
        gi = f
    gi = min(gi, g.frame_count - 1)
    if cursor_state.get("i") != gi:
        img = g.read(gi)
        cursor_state["i"] = gi
        cursor_state["img"] = M.resize(img.max(axis=2), run.width, run.height, "linear")
    return cursor_state["img"]


def assemble(run, progress=None, preview_max=96, preview_w=960):
    t0 = time.time()
    cfg = run.cfg
    n = run.n
    weights = _tile_weights(run)
    os.makedirs(run.out_dir, exist_ok=True)
    if run.rgba_dir:
        os.makedirs(run.rgba_dir, exist_ok=True)
    k_prev = min(preview_max, n)
    sample = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, k_prev)))
    sample_set = set(sample)
    need_guided = (run.work_w != run.width or run.work_h != run.height) and cfg["guided_upscale"]
    need_plate = [f for f in range(n) if need_guided or run.rgba_dir or f in sample_set]
    plate = run.plate()
    plate_iter = plate.iter_frames([run.start + f for f in need_plate]) if need_plate else iter(())
    need_set = set(need_plate)
    digits = max(4, len(str(max(abs(x) for x in run.out_numbers))))
    pv_w = min(preview_w, run.width)
    pv_h = max(2, int(round(run.height * pv_w / float(run.width))))
    pv_h += pv_h % 2
    mattes, comps = [], []
    stats = {"band": 0.0, "tile_frames": 0, "coverage": 0.0}
    gstate = {}
    written = []
    chk = M.checker(pv_h, pv_w, size=max(8, pv_w // 60))
    for f in range(n):
        disp = None
        if f in need_set:
            _, raw = next(plate_iter)
            disp = run.display(raw)
        A = assemble_frame(run, f, weights, plate_display=disp if need_guided else None, stats=stats)
        g = _garbage_frame(run, f, gstate)
        if g is not None:
            A = A * np.clip(g, 0.0, 1.0)
        A = M.clip_remap(A, float(cfg["black_clip"]), float(cfg["white_clip"]))
        if cfg["invert"]:
            A = 1.0 - A
        num = run.out_numbers[f]
        name = f"{run.base_name}_matte.{num:0{digits}d}.{run.out_ext}" if num >= 0 else \
            f"{run.base_name}_matte.-{abs(num):0{digits}d}.{run.out_ext}"
        path = os.path.join(run.out_dir, name)
        write_matte(path, A, run.fmt)
        written.append(path)
        if run.rgba_dir:
            write_rgba16(os.path.join(run.rgba_dir, f"{run.base_name}_rgba.{num:0{digits}d}.png"), disp, A)
        stats["coverage"] += float(A.mean())
        if f in sample_set:
            am = M.resize(A, pv_w, pv_h, "area")
            mattes.append(np.repeat(am[:, :, None], 3, axis=2))
            dp = M.resize(disp, pv_w, pv_h, "area")
            comps.append(dp * am[:, :, None] + chk * (1.0 - am[:, :, None]))
        if progress:
            progress(f + 1, n)
    run.close()
    secs = time.time() - t0
    detail_written = len(run.written.get("detail", []))
    jobs = run.detail_jobs
    report = "\n".join([
        f"Wrote {len(written)} frames, {run.width}x{run.height}, {cfg['output_format']}, "
        f"{os.path.basename(written[0])} .. {os.path.basename(written[-1])}",
        f"Folder: {run.out_dir}" + (f"  (+ RGBA in {run.rgba_dir})" if run.rgba_dir else ""),
        f"Proxy: {len(run.chunks)} chunk(s) at {run.geom['win_w']}x{run.geom['win_h']}",
        f"Detail: {len(jobs)} jobs ({len(set(j['tile'] for j in jobs))} tiles x chunks) at {run.work_w}x{run.work_h}, "
        f"{detail_written} tile frames stored"
        + (f" - none because {run.detail_reason}" if getattr(run, 'detail_reason', None) else ""),
        f"Band coverage (proxy): {100 * stats['band'] / n:.2f}%   Mean matte coverage: {100 * stats['coverage'] / n:.2f}%",
        f"Assemble time: {secs:.1f}s ({secs / n:.2f}s/frame); total since Setup {time.time() - run.t_start:.1f}s",
    ] + ([f"Warnings ({len(run.warnings)}): " + "; ".join(run.warnings[:5])] if run.warnings else []))
    run.report = report
    if not cfg["keep_cache"]:
        shutil.rmtree(run.cache_dir, ignore_errors=True)
    else:
        run.save_json()
    mp = np.stack(mattes).astype(np.float32)
    cp = np.stack(comps).astype(np.float32)
    return mp, cp, run.out_dir, report, written
