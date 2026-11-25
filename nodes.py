"""ComfyUI wrappers for i2 Roto. Thin: all logic is in i2roto/*.py."""
import os
import logging

import numpy as np
import torch

from .i2roto import pipeline as PL
from .i2roto import plan as P
from .i2roto.color import COLOR_MODES
from .i2roto.io_utils import I2RotoError

log = logging.getLogger("i2roto")
CATEGORY = "i2/Roto"


class AnyType(str):
    """Matches any type (ComfyUI convention for '*' inputs)."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")


def _dirs():
    try:
        import folder_paths  # noqa: WPS433
        return folder_paths.get_temp_directory(), folder_paths.get_output_directory()
    except Exception:  # running outside ComfyUI (tests)
        base = os.environ.get("I2ROTO_TEST_DIR") or os.path.join(os.getcwd(), "i2roto_test")
        return os.path.join(base, "temp"), os.path.join(base, "output")


def _progress(total):
    try:
        import comfy.utils  # noqa: WPS433
        bar = comfy.utils.ProgressBar(total)
        return lambda i, n: bar.update_absolute(i, n)
    except Exception:
        return None


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _np(img):
    if isinstance(img, torch.Tensor):
        return img.detach().cpu().float().numpy()
    return np.asarray(img, dtype=np.float32)


def _first(x):
    return x[0] if isinstance(x, list) else x


def _wrap(fn):
    """Turn I2RotoError into a plain RuntimeError message (ComfyUI shows the message)."""
    try:
        return fn()
    except I2RotoError as e:
        raise RuntimeError(f"i2 Roto: {e}") from None


class I2RotoSetup:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plate_path": ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "Video file, folder of frames, or sequence (shot.####.exr / shot.%04d.exr / any one frame)."}),
            "start_frame": ("INT", {"default": 0, "min": 0, "max": 10_000_000, "tooltip": "Offset into the clip (0 = first frame)."}),
            "frame_count": ("INT", {"default": 0, "min": 0, "max": 10_000_000, "tooltip": "0 = to the end."}),
            "plate_color": (COLOR_MODES, {"default": COLOR_MODES[0]}),
            "window": (P.WINDOW_NAMES, {"default": P.DEFAULT_WINDOW,
                                        "tooltip": "Model working window. 1920x1088 is what the model was trained at."}),
            "chunk_frames": ("INT", {"default": 49, "min": 9, "max": 145,
                                     "tooltip": "Frames per model pass, snapped to 8n+1 (9..145)."}),
            "detail": (P.DETAIL_MODES, {"default": P.DETAIL_NATIVE}),
            "tile_overlap": ("INT", {"default": 256, "min": 0, "max": 1024, "step": 8}),
            "band_px": ("INT", {"default": 48, "min": 2, "max": 1024,
                                "tooltip": "Half-width of the edge band in native plate pixels."}),
            "tile_trust": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "pin_chunks": ("BOOLEAN", {"default": True}),
            "black_clip": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 1.0, "step": 0.001}),
            "white_clip": ("FLOAT", {"default": 0.995, "min": 0.0, "max": 1.0, "step": 0.001}),
            "invert": ("BOOLEAN", {"default": False}),
            "edge_shift_px": ("FLOAT", {"default": 0.0, "min": -64.0, "max": 64.0, "step": 0.25,
                                        "tooltip": "Negative = choke, positive = spread (native px)."}),
            "guided_upscale": ("BOOLEAN", {"default": True}),
            "garbage_matte_path": ("STRING", {"default": "", "tooltip": "Optional. White = keep."}),
            "output_folder": ("STRING", {"default": "", "tooltip": "Empty = ComfyUI/output/i2roto/<plate>_matte. Relative = under ComfyUI/output."}),
            "output_format": (PL.OUTPUT_FORMAT_NAMES, {"default": "PNG 16-bit"}),
            "write_rgba": ("BOOLEAN", {"default": False}),
            "start_number": ("INT", {"default": 1001, "min": -1_000_000, "max": 10_000_000}),
            "keep_cache": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("I2ROTO_RUN", "INT", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("run", "proxy_jobs", "blank_pin", "first_frame", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, **kwargs):
        temp, out = _dirs()
        run, jobs, blank, first, info = _wrap(lambda: PL.setup(kwargs, temp, out))
        log.info("[i2 Roto] " + info.replace("\n", "\n[i2 Roto] "))
        return (run, jobs, _t(blank), _t(first), info)


class I2RotoLoadJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "run": ("I2ROTO_RUN",),
            "stage": (["proxy", "detail"], {"default": "proxy"}),
            "job_index": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "BOOLEAN", "I2ROTO_JOB")
    RETURN_NAMES = ("images", "width", "height", "length", "is_first_chunk", "job")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load(self, run, stage, job_index):
        imgs, job = _wrap(lambda: PL.load_job(run, stage, int(job_index)))
        job = dict(job)
        job["run"] = run
        return (_t(imgs), int(job["win_w"]), int(job["win_h"]), int(job["length"]), bool(job["is_first_chunk"]), job)


class I2RotoStoreJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": ("I2ROTO_JOB",), "matte": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "I2ROTO_RUN")
    RETURN_NAMES = ("next_pin", "done")
    FUNCTION = "store"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def store(self, job, matte):
        run = job["run"]
        pin, run = _wrap(lambda: PL.store_job(run, job, _np(matte)))
        return (_t(pin), run)


class I2RotoPlanDetail:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run": ("I2ROTO_RUN",), "proxy_done": (ANY,)}}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("I2ROTO_RUN", "INT", "STRING")
    RETURN_NAMES = ("run", "detail_jobs", "info")
    FUNCTION = "plan"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def plan(self, run, proxy_done=None):
        run = _first(run)
        run, n, info = _wrap(lambda: PL.plan_detail(run))
        log.info("[i2 Roto] " + info)
        return (run, n, info)


class I2RotoAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run": ("I2ROTO_RUN",), "detail_done": (ANY,)}}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("matte_preview", "comp_preview", "output_folder", "report")
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def assemble(self, run, detail_done=None):
        run = _first(run)
        n_done = len(detail_done) if isinstance(detail_done, list) else -1
        log.info(f"[i2 Roto] Assemble: detail loop returned {n_done} item(s)")
        mp, cp, folder, report, _ = _wrap(lambda: PL.assemble(run, progress=_progress(run.n)))
        report = report + f"\nDetail loop returned {n_done} item(s) to Assemble"
        log.info("[i2 Roto] " + report.replace("\n", "\n[i2 Roto] "))
        return {"ui": {"text": [report]}, "result": (_t(mp), _t(cp), folder, report)}


class I2RotoStubMatte:
    """DEV ONLY: returns the synthetic test plate's ground-truth alpha for the job (no model)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "job": ("I2ROTO_JOB",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matte",)
    FUNCTION = "stub"
    CATEGORY = CATEGORY + "/dev"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def stub(self, images, job):
        from .i2roto.oracle import oracle_job  # noqa: WPS433
        m = _wrap(lambda: oracle_job(job["run"], job))
        if m.shape[0] != images.shape[0] or m.shape[1] != images.shape[1] or m.shape[2] != images.shape[2]:
            raise RuntimeError(f"stub matte shape {m.shape} does not match images {tuple(images.shape)}")
        return (_t(np.repeat(m[..., None], 3, axis=3)),)


NODE_CLASS_MAPPINGS = {
    "I2RotoSetup": I2RotoSetup,
    "I2RotoLoadJob": I2RotoLoadJob,
    "I2RotoStoreJob": I2RotoStoreJob,
    "I2RotoPlanDetail": I2RotoPlanDetail,
    "I2RotoAssemble": I2RotoAssemble,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "I2RotoSetup": "i2 Roto · Setup",
    "I2RotoLoadJob": "i2 Roto · Load Job",
    "I2RotoStoreJob": "i2 Roto · Store Job",
    "I2RotoPlanDetail": "i2 Roto · Plan Detail Pass",
    "I2RotoAssemble": "i2 Roto · Assemble + Write",
}
if os.environ.get("I2ROTO_DEV") == "1":
    NODE_CLASS_MAPPINGS["I2RotoStubMatte"] = I2RotoStubMatte
    NODE_DISPLAY_NAME_MAPPINGS["I2RotoStubMatte"] = "i2 Roto · Stub Matte (DEV)"
