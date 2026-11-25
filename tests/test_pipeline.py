"""Test 1: pipeline without ComfyUI. Drives the node functions directly with an oracle matte
(ground truth of the synthetic plate for each job's region/frames).

    python tests/test_pipeline.py [section ...]      sections: chunks options video sixk seams exr
    (default: all except exr; exr needs an OpenCV build with OpenEXR, see RESULTS.md)

Data goes to $I2ROTO_WORK (default /home/claude/i2roto/_work). Results are printed and saved as JSON.
"""
import json
import os
import shutil
import sys
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

WORK = os.environ.get("I2ROTO_WORK", "/home/claude/i2roto/_work")
os.environ["I2ROTO_TEST_DIR"] = os.path.join(WORK, "pt")
import harness  # noqa: E402
import plate_gen  # noqa: E402
from i2roto.synth import Synth  # noqa: E402
from i2roto import merge as M  # noqa: E402

RESULTS = {}
FAIL = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    RESULTS.setdefault("checks", []).append({"name": name, "pass": bool(cond), "detail": detail})
    if not cond:
        FAIL.append(name)


def clipped_gt(g, b=0.005, w=0.995):
    return np.clip((g - b) / (w - b), 0, 1)


def out_files(folder, ext):
    return sorted(f for f in os.listdir(folder) if f.endswith("." + ext))


def read_out(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return a


def ensure_seq(name, w, h, n, start_number=1001, bits=16):
    folder = os.path.join(WORK, "data", name)
    meta = os.path.join(folder, "i2roto_synth.json")
    if os.path.exists(meta) and len([f for f in os.listdir(folder) if f.endswith(".png")]) == n:
        return folder, Synth.load(meta)
    shutil.rmtree(folder, ignore_errors=True)
    t = time.time()
    s, _ = plate_gen.write_png_sequence(folder, w, h, n, start_number=start_number, bits=bits)
    print(f"  generated {name}: {w}x{h} x{n} in {time.time() - t:.1f}s")
    return folder, s


def base_cfg(plate, out, **kw):
    cfg = dict(plate_path=plate, output_folder=out, keep_cache=False)
    cfg.update(kw)
    return cfg


# ============================================================================ chunks
def test_chunks():
    print("\n=== chunking: n = 20, 49, 50, 97, 130 with chunk_frames 49 (1920x1088 plate) ===")
    folder, synth = ensure_seq("chunk1088", 1920, 1088, 137, bits=8)
    res = {}
    for n, start in [(20, 0), (49, 0), (50, 0), (97, 0), (130, 0), (50, 7)]:
        out = os.path.join(WORK, "out", f"chunks_{n}_{start}")
        shutil.rmtree(out, ignore_errors=True)
        r = harness.run_graph(base_cfg(os.path.join(folder, "shot.####.png"), out, frame_count=n, start_frame=start,
                                       chunk_frames=49, window="1920x1088 (model native, best)",
                                       start_number=5), verbose=False)
        run = r["run"]
        written = run.written["proxy"]
        files = out_files(out, "png")
        expect_names = [f"shot_matte.{1001 + start + i:04d}.png" for i in range(n)]
        if n <= 49:
            exp_chunks = 1
        else:
            exp_chunks = -(-(n - 1) // 48)
        lengths = [c["length"] for c in run.chunks]
        errs, wrong = [], 0
        for i in range(n):
            a = read_out(os.path.join(out, expect_names[i])).astype(np.float32) / 65535
            g = clipped_gt(synth.alpha(start + i))
            e = float(np.abs(a - g).mean())
            errs.append(e)
            # identity: closer to its own frame than to the neighbours
            # (skipped where the neighbour's ground truth is identical, e.g. the figure has left the frame)
            for j in (start + i - 1, start + i + 1):
                if 0 <= j < synth.N:
                    gj = clipped_gt(synth.alpha(j))
                    if float(np.abs(gj - g).mean()) > 1e-5 and float(np.abs(a - gj).mean()) <= e:
                        wrong += 1
        ok = (sorted(written) == list(range(n)) and len(written) == n and files == expect_names
              and len(run.chunks) == exp_chunks and wrong == 0 and max(errs) < 1e-4 and not run.warnings
              and r["flags"]["proxy"] == [True] + [False] * (exp_chunks - 1))
        detail = (f"chunks {len(run.chunks)} (expect {exp_chunks}) lengths {lengths}; frames stored {len(written)} "
                  f"unique {len(set(written))}; files {len(files)} {files[0]}..{files[-1]}; max frame MAE "
                  f"{max(errs):.2e}; identity violations {wrong}; is_first flags {r['flags']['proxy']}")
        check(f"chunks n={n} start={start}", ok, detail)
        res[f"{n}_{start}"] = detail
    RESULTS["chunks"] = res


# ============================================================================ options
def test_options():
    print("\n=== options smoke: TIFF, invert, edge shift, garbage matte, RGBA, log colour, 2x proxy ===")
    folder, synth = ensure_seq("opt1536", 1536, 864, 12)
    gm = os.path.join(WORK, "data", "garbage_left.png")
    g = np.zeros((864, 1536), np.uint8)
    g[:, :768] = 255
    cv2.imwrite(gm, g)
    out = os.path.join(WORK, "out", "options")
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(out + "_rgba", ignore_errors=True)
    r = harness.run_graph(base_cfg(os.path.join(folder, "shot.1001.png"), out, output_format="TIFF 16-bit",
                                   invert=True, edge_shift_px=2.5, garbage_matte_path=f'"{gm}"', write_rgba=True,
                                   plate_color="log -> auto normalize", detail="2x proxy", chunk_frames=9),
                          verbose=False)
    files = out_files(out, "tif")
    a = read_out(os.path.join(out, files[3]))
    rgba = sorted(os.listdir(out + "_rgba"))
    x = cv2.imread(os.path.join(out + "_rgba", rgba[3]), cv2.IMREAD_UNCHANGED)
    gt = synth.alpha(3)
    inv = 1 - a.astype(np.float32) / 65535
    right_zero = float(inv[:, 800:].max())
    check("options: 12 TIFF files, uint16, 1536x864", len(files) == 12 and a.dtype == np.uint16 and a.shape == (864, 1536),
          f"{len(files)} files, {a.dtype}, {a.shape}")
    check("options: garbage matte kills right half, invert applied", right_zero < 1e-6 and float(inv[:, :700].max()) > 0.99,
          f"max(1-out) right half {right_zero}, left max {float(inv[:, :700].max()):.3f}")
    spread = float((inv[:, :760] > 0.5).sum()) / max(1.0, float((gt[:, :760] > 0.5).sum()))
    check("options: edge spread 2.5 px grows the matte", spread > 1.0, f"area ratio {spread:.4f}")
    check("options: RGBA 16-bit PNG written", len(rgba) == 12 and x.dtype == np.uint16 and x.shape == (864, 1536, 4),
          f"{len(rgba)} files {x.dtype} {x.shape}")
    check("options: 2x proxy working res", r["run"].work_w == 1536 and r["run"].d == 1.0,
          f"plate 1536 wide fits 2x proxy: work {r['run'].work_w}x{r['run'].work_h}, d={r['run'].d}")
    RESULTS["options"] = {"log_params": r["run"].log_params, "report": r["report"]}


# ============================================================================ video
def test_video():
    print("\n=== video input == sequence input (6144x3240, 20 frames) ===")
    folder, synth = ensure_seq("sixk", 6144, 3240, 20)
    pack = harness.load_pack()
    PL = pack.i2roto.pipeline if hasattr(pack, "i2roto") else None
    from i2roto_pack.i2roto import pipeline as PL  # noqa
    vids = {}
    for codec, ext in [("prores", "mov"), ("h264", "mp4")]:
        p = os.path.join(WORK, "data", "sixk_video", f"shot_{codec}.{ext}")
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            t = time.time()
            plate_gen.write_video(p, 6144, 3240, 20, codec=codec, gop=8)
            print(f"  encoded {p} in {time.time() - t:.1f}s ({os.path.getsize(p) / 1e6:.0f} MB)")
        vids[codec] = p
    res = {}
    for codec, p in vids.items():
        tmp, outd = os.path.join(WORK, "pt", "temp"), os.path.join(WORK, "pt", "output")
        cfgv = dict(plate_path=p, start_frame=11, frame_count=6, chunk_frames=9, detail="native (max quality)")
        cfgs = dict(plate_path=os.path.join(folder, "shot.####.png"), start_frame=11, frame_count=6, chunk_frames=9)
        rv, *_ = PL.setup(cfgv, tmp, outd)
        rs, *_ = PL.setup(cfgs, tmp, outd)
        check(f"video {codec}: probe", rv.width == 6144 and rv.height == 3240 and rv.total_frames == 20 and
              abs(rv.fps - 24) < 1e-6, f"{rv.width}x{rv.height}, {rv.total_frames} frames, {rv.fps} fps")
        iv, _ = PL.load_job(rv, "proxy", 0)
        is_, _ = PL.load_job(rs, "proxy", 0)
        mae = float(np.abs(iv - is_).mean())
        # Frame identity on the decoded proxy job (source frames 11..16, padded to 9).
        own = [float(np.abs(iv[k] - is_[k]).mean()) for k in range(6)]
        nb = [float(np.abs(iv[k] - is_[k + 1]).mean()) for k in range(5)]
        # Random access across GOPs at native res.
        idx = [0, 5, 7, 8, 13, 19, 19]
        vp, sp = rv.plate(), rs.plate()
        diffs = []
        for (i1, a1), (i2, a2) in zip(vp.iter_frames(idx), sp.iter_frames(idx)):
            diffs.append(float(np.abs(a1 - a2).mean()))
        # A frame read after a separate seek (no shared decoder state)
        single = float(np.abs(vp.read(13) - sp.read(13)).mean())
        tol = 0.01 if codec == "prores" else 0.02
        ok = mae < tol and max(own) < min(nb) and max(diffs) < tol and single < tol
        d = (f"proxy job MAE {mae:.5f}; own-frame MAE max {max(own):.5f} < neighbour-frame MAE min {min(nb):.5f}; "
             f"native random-access MAE per frame {[round(x, 5) for x in diffs]}; single seek f13 {single:.5f}; tol {tol}")
        check(f"video {codec} matches sequence", ok, d)
        res[codec] = d
        rv.close(); rs.close()
        shutil.rmtree(rv.cache_dir, ignore_errors=True); shutil.rmtree(rs.cache_dir, ignore_errors=True)
    RESULTS["video"] = res


# ============================================================================ 6K
def eval_run(out, synth, n, band_dilate=24, tiles=None):
    """Streaming error stats vs native ground truth."""
    files = out_files(out, "png")
    k = M._disk(band_dilate)
    tot = {"mae_raw": 0.0, "mae_clip": 0.0, "band_mae": 0.0, "band_px": 0, "line_sum": 0.0, "line_px": 0,
           "line_max": 0.0, "band_max": 0.0}
    for i, f in enumerate(files):
        a = read_out(os.path.join(out, f)).astype(np.float32) / 65535
        g = synth.alpha(i)
        gc = clipped_gt(g)
        e = np.abs(a - gc)
        tot["mae_raw"] += float(np.abs(a - g).mean())
        tot["mae_clip"] += float(e.mean())
        core = ((g > 0.002) & (g < 0.998)).astype(np.uint8)
        band = cv2.dilate(core, k) > 0
        tot["band_mae"] += float(e[band].sum())
        tot["band_px"] += int(band.sum())
        tot["band_max"] = max(tot["band_max"], float(e[band].max()))
        if tiles:
            lm = np.zeros_like(band)
            for t in tiles:
                for u in tiles:
                    if u["y"] == t["y"] and u["x"] > t["x"] and u["x"] < t["x"] + t["w"]:
                        xc = (u["x"] + t["x"] + t["w"]) // 2
                        lm[t["y"]:t["y"] + t["h"], xc] = True
                    if u["x"] == t["x"] and u["y"] > t["y"] and u["y"] < t["y"] + t["h"]:
                        yc = (u["y"] + t["y"] + t["h"]) // 2
                        lm[yc, t["x"]:t["x"] + t["w"]] = True
            sel = lm & band
            if sel.any():
                tot["line_sum"] += float(e[sel].sum())
                tot["line_px"] += int(sel.sum())
                tot["line_max"] = max(tot["line_max"], float(e[sel].max()))
    nf = len(files)
    return {"frames": nf, "mae_raw": tot["mae_raw"] / nf, "mae_clip": tot["mae_clip"] / nf,
            "band_mae": tot["band_mae"] / max(1, tot["band_px"]), "band_max": tot["band_max"],
            "line_mae": tot["line_sum"] / max(1, tot["line_px"]), "line_max": tot["line_max"],
            "line_px": tot["line_px"]}


def test_sixk():
    print("\n=== 6144x3240 x 20 frames, 16-bit PNG sequence: native / 2x proxy / proxy only ===")
    folder, synth = ensure_seq("sixk", 6144, 3240, 20)
    res = {}
    for mode in ["native (max quality)", "2x proxy", "proxy only (fastest)"]:
        key = mode.split()[0]
        out = os.path.join(WORK, "out", f"sixk_{key}")
        shutil.rmtree(out, ignore_errors=True)
        t0 = time.time()
        r = harness.run_graph(base_cfg(os.path.join(folder, "shot.####.png"), out, detail=mode), verbose=True)
        wall = time.time() - t0
        run = r["run"]
        files = out_files(out, "png")
        a0 = read_out(os.path.join(out, files[0]))
        ev = eval_run(out, synth, 20, tiles=run.detail_tiles if key == "native" else None)
        res[key] = {"wall_s": wall, "timings": r["timings"], "assemble_s_per_frame": r["timings"]["assemble"] / 20,
                    "tiles": len(run.detail_tiles), "jobs": r["detail_jobs"], "eval": ev,
                    "files": [files[0], files[-1], len(files)], "dtype": str(a0.dtype), "shape": a0.shape,
                    "report": r["report"], "plan": r["plan_info"]}
        print(json.dumps(res[key]["eval"]))
        check(f"6K {key}: 20 files shot_matte.1001..1020.png, uint16 single-channel 6144x3240",
              files == [f"shot_matte.{1001 + i}.png" for i in range(20)] and a0.dtype == np.uint16
              and a0.shape == (3240, 6144), f"{len(files)} files {files[0]}..{files[-1]} {a0.dtype} {a0.shape}")
        if key == "native":
            check("6K native: MAE vs native ground truth < 0.004", ev["mae_raw"] < 0.004,
                  f"MAE raw GT {ev['mae_raw']:.6f}, vs clip-remapped GT {ev['mae_clip']:.6f}")
    nat, prx, two = res["native"]["eval"], res["proxy"]["eval"], res["2x"]["eval"]
    check("6K: native edge-band MAE clearly better than proxy only (< 0.5x)",
          nat["band_mae"] < 0.5 * prx["band_mae"],
          f"band MAE native {nat['band_mae']:.6f} / 2x proxy {two['band_mae']:.6f} / proxy only {prx['band_mae']:.6f}")
    check("6K: seams - error on tile-overlap centre lines no worse than the band",
          nat["line_mae"] <= max(nat["band_mae"], 2e-5) and nat["line_max"] <= nat["band_max"] + 1e-6,
          f"centre-line MAE {nat['line_mae']:.2e} (max {nat['line_max']:.2e}, {nat['line_px']} px) vs band MAE "
          f"{nat['band_mae']:.2e} (band max {nat['band_max']:.2e})")
    RESULTS["sixk"] = res


# ============================================================================ seam stress
def test_seams():
    """Tiles that DISAGREE. Proxy oracle = flat 0.5 (so the edge band covers the whole frame and the tile grid
    covers everything); each detail tile returns a flat value, 0.45 or 0.55 by tile index parity. The written
    matte inside the band is then exactly the tile blend, so any seam shows up as a step between adjacent pixels.
    A hard (unblended) seam would be a 0.10 step; the smoothstep ramp over 256 px should give < ~0.001/px."""
    print("\n=== seam stress: disagreeing flat tiles (3072x1620) ===")
    folder, synth = ensure_seq("seam3k", 3072, 1620, 9)

    def flat(imgs, job):
        import torch
        if job["stage"] == "proxy":
            v = 0.5
        else:
            v = 0.55 if job["tile"] % 2 == 0 else 0.45
        return (torch.full(tuple(imgs.shape), v, dtype=torch.float32),)
    out = os.path.join(WORK, "out", "seams")
    shutil.rmtree(out, ignore_errors=True)
    r = harness.run_graph(base_cfg(os.path.join(folder, "shot.####.png"), out, chunk_frames=9, black_clip=0.0,
                                   white_clip=1.0, window="1280x704 (low VRAM, faster)"), verbose=False, matte_fn=flat)
    tiles = r["run"].detail_tiles
    a = read_out(os.path.join(out, out_files(out, "png")[4])).astype(np.float32) / 65535
    step = max(float(np.abs(np.diff(a, axis=1)).max()), float(np.abs(np.diff(a, axis=0)).max()))
    # Interior of each tile (outside all overlaps) must equal that tile's value.
    worst_interior = 0.0
    for t in tiles:
        v = 0.55 if t["index"] % 2 == 0 else 0.45
        x0, x1 = t["x"] + t["ox"], t["x"] + t["w"] - t["ox"]
        y0, y1 = t["y"] + t["oy"], t["y"] + t["h"] - t["oy"]
        mask = np.ones(a.shape, bool)
        for u in tiles:
            if u is not t:
                mask[u["y"]:u["y"] + u["h"], u["x"]:u["x"] + u["w"]] = False
        reg = mask[y0:y1, x0:x1]
        if reg.any():
            worst_interior = max(worst_interior, float(np.abs(a[y0:y1, x0:x1][reg] - v).max()))
    lo, hi = float(a.min()), float(a.max())
    check("seam stress: no visible step where disagreeing tiles meet (max adjacent-pixel step < 0.002; hard seam = 0.10)",
          step < 0.002, f"max adjacent-pixel step {step:.5f}; {len(tiles)} tiles; output range {lo:.3f}..{hi:.3f}")
    check("seam stress: tile interiors keep their own value", worst_interior < 1e-3,
          f"max interior deviation {worst_interior:.2e}")
    RESULTS["seams"] = {"max_step": step, "tiles": len(tiles), "interior_dev": worst_interior}


# ============================================================================ EXR
def test_exr():
    print("\n=== EXR: linear EXR plate in, EXR half out ===")
    import i2roto.io_utils as IO
    cv = IO.get_cv2()
    if not cv.haveImageWriter("x.exr"):
        print("  this OpenCV has no OpenEXR: testing only the built-in EXR writer + the clear error on EXR input")
        folder = os.path.join(WORK, "data", "exr_none")
        os.makedirs(folder, exist_ok=True)
        open(os.path.join(folder, "shot.0001.exr"), "wb").write(b"\x76\x2f\x31\x01")
        try:
            harness.run_graph(base_cfg(os.path.join(folder, "shot.####.exr"), os.path.join(WORK, "out", "x")),
                              verbose=False)
            check("EXR input without OpenEXR gives a clear error", False, "no error raised")
        except RuntimeError as e:
            check("EXR input without OpenEXR gives a clear error", "EXR" in str(e), str(e))
        # EXR output end to end through the built-in writer; verified by a reader with OpenEXR if one is given.
        folder, synth = ensure_seq("opt1536", 1536, 864, 12)
        out = os.path.join(WORK, "out", "exr_builtin_run")
        shutil.rmtree(out, ignore_errors=True)
        harness.run_graph(base_cfg(os.path.join(folder, "shot.####.png"), out, output_format="EXR half float",
                                   frame_count=3, chunk_frames=9), verbose=False)
        files = out_files(out, "exr")
        check("EXR out via built-in writer: 3 files written", len(files) == 3, str(files))
        reader = os.environ.get("I2ROTO_EXR_READER_PATH")
        if reader:
            import subprocess
            np.save(os.path.join(out, "gt.npy"), clipped_gt(synth.alpha(1)))
            code = ("import os,sys,numpy as np;os.environ['OPENCV_IO_ENABLE_OPENEXR']='1';import cv2;"
                    f"a=cv2.imread({os.path.join(out, files[1])!r},-1);g=np.load({os.path.join(out, 'gt.npy')!r});"
                    "print(a.dtype,a.shape,float(abs(a-g).mean()))")
            p = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=reader),
                               capture_output=True, text=True)
            parts = p.stdout.split()
            ok = p.returncode == 0 and parts[0] == "float32" and float(parts[-1]) < 0.004
            check("EXR out via built-in writer reads back in an OpenEXR-enabled OpenCV, MAE < 0.004", ok,
                  p.stdout.strip() + p.stderr.strip()[-300:])
        RESULTS["exr"] = {"opencv_exr": False}
        return
    folder = os.path.join(WORK, "data", "exr1536")
    if not os.path.exists(os.path.join(folder, "shot.0006.exr")):
        plate_gen.write_exr_sequence(folder, 1536, 864, 6)
    synth = Synth.load(os.path.join(folder, "i2roto_synth.json"))
    out = os.path.join(WORK, "out", "exr")
    shutil.rmtree(out, ignore_errors=True)
    r = harness.run_graph(base_cfg(os.path.join(folder, "shot.0003.exr"), out, plate_color="linear (EXR) -> sRGB",
                                   output_format="EXR half float", window="1280x704 (low VRAM, faster)",
                                   chunk_frames=9), verbose=False)
    files = out_files(out, "exr")
    a = cv.imread(os.path.join(out, files[2]), cv.IMREAD_UNCHANGED)
    g = clipped_gt(synth.alpha(2))
    mae = float(np.abs(a - g).mean())
    check("EXR: 6 EXR half files, single channel float, numbers kept (0001..0006)",
          files == [f"shot_matte.{i:04d}.exr" for i in range(1, 7)] and a.ndim == 2 and a.dtype == np.float32,
          f"{files[0]}..{files[-1]} ndim {a.ndim} {a.dtype} {a.shape}")
    check("EXR: matte round-trip MAE vs ground truth < 0.004", mae < 0.004, f"MAE {mae:.6f}")
    # linear EXR -> display conversion equals the display-space plate
    from i2roto_pack.i2roto import pipeline as PL
    run = r["run"]
    disp = PL.to_display(cv.imread(os.path.join(folder, "shot.0002.exr"), cv.IMREAD_UNCHANGED)[:, :, ::-1],
                         "linear (EXR) -> sRGB") if hasattr(PL, "to_display") else None
    ref = synth.plate(1)
    dmae = float(np.abs(disp - ref).mean())
    check("EXR: linear plate -> sRGB matches the display plate", dmae < 0.002, f"MAE {dmae:.6f}")
    # built-in writer, read back by OpenCV's EXR reader
    p = os.path.join(WORK, "out", "exr_builtin.exr")
    IO.write_exr_half_y(p, g)
    b = cv.imread(p, cv.IMREAD_UNCHANGED)
    check("EXR: built-in writer (used when OpenCV lacks OpenEXR) reads back in OpenCV",
          b is not None and b.shape == g.shape and float(np.abs(b - g).max()) < 1e-3,
          f"max err {float(np.abs(b - g).max()):.2e}")
    RESULTS["exr"] = {"opencv_exr": True, "cv2": cv.__version__, "mae": mae}


def test_guided():
    """Informational + sanity: proxy-only upscale with and without the guided filter, vs native ground truth."""
    print("\n=== guided upscale on/off (proxy only, 3072x1620) ===")
    folder, synth = ensure_seq("seam3k", 3072, 1620, 9)
    res = {}
    for g in (True, False):
        out = os.path.join(WORK, "out", f"guided_{g}")
        shutil.rmtree(out, ignore_errors=True)
        harness.run_graph(base_cfg(os.path.join(folder, "shot.####.png"), out, detail="proxy only (fastest)",
                                   window="1280x704 (low VRAM, faster)", chunk_frames=9, guided_upscale=g),
                          verbose=False)
        res[g] = eval_run(out, synth, 9)
    d = (f"band MAE guided {res[True]['band_mae']:.5f} vs bicubic {res[False]['band_mae']:.5f}; "
         f"frame MAE guided {res[True]['mae_raw']:.6f} vs bicubic {res[False]['mae_raw']:.6f}")
    check("guided upscale is not worse than plain bicubic in the edge band", res[True]["band_mae"] <= res[False]["band_mae"], d)
    RESULTS["guided"] = {str(k): v for k, v in res.items()}


SECTIONS = {"guided": test_guided, "chunks": test_chunks, "options": test_options, "video": test_video, "sixk": test_sixk,
            "seams": test_seams, "exr": test_exr}

if __name__ == "__main__":
    names = sys.argv[1:] or ["chunks", "options", "video", "seams", "sixk", "exr"]
    t0 = time.time()
    for n in names:
        SECTIONS[n]()
    RESULTS["failed"] = FAIL
    RESULTS["seconds"] = time.time() - t0
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    tag = "_".join(names)
    with open(os.path.join(WORK, "results", f"pipeline_{tag}.json"), "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)
    print(f"\n{len(FAIL)} failed: {FAIL}  ({time.time() - t0:.0f}s)")
    sys.exit(1 if FAIL else 0)
