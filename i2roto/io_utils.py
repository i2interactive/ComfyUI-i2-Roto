"""File and path helpers: image read/write and frame-sequence discovery.

Imports without ComfyUI. cv2 is imported lazily (get_cv2) so a missing OpenCV gives a clear error.
"""
import os
import re
import glob

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

_cv2 = None

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".dpx")
VIDEO_EXTS = (".mov", ".mp4", ".m4v", ".mkv", ".mxf", ".avi", ".webm", ".mts", ".m2ts", ".ts", ".mpg", ".mpeg")


class I2RotoError(Exception):
    """Plain-English error shown to the user."""


def get_cv2():
    global _cv2
    if _cv2 is None:
        try:
            import cv2  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            raise I2RotoError(
                "i2 Roto needs OpenCV. Install it into ComfyUI's Python with: "
                "python -m pip install opencv-python"
            ) from e
        _cv2 = cv2
    return _cv2


def clean_path(p):
    """Strip whitespace and surrounding quotes, accept / and \\ on any OS, expand ~."""
    if p is None:
        return ""
    p = str(p).strip()
    while len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'":
        p = p[1:-1].strip()
    if not p:
        return ""
    if os.sep == "/":
        p = p.replace("\\", "/")
    else:
        p = p.replace("/", os.sep)
    p = os.path.expanduser(p)
    return os.path.normpath(p)


# ----------------------------------------------------------------------------- image IO

def _check_ext_supported(path, write=False):
    cv2 = get_cv2()
    ext = os.path.splitext(path)[1].lower()
    ok = cv2.haveImageWriter(path) if write else True
    if ext == ".exr" and not cv2.haveImageWriter("x.exr"):
        raise I2RotoError(
            "This OpenCV build has no OpenEXR support, so EXR plates cannot be read (the opencv-python 5.x pip "
            "wheels are built without it; the 4.x wheels have it). Convert the plate to a 16-bit TIFF/PNG "
            "sequence, or use an OpenCV with OpenEXR. (EXR output still works: the pack has its own EXR writer.)"
        )
    if ext == ".dpx" and not cv2.haveImageWriter("x.dpx"):
        raise I2RotoError("This OpenCV build cannot read DPX files. Convert the plate to EXR, TIFF or PNG first.")
    return ok


def read_image(path):
    """Read an image file as float32 HxWx3 RGB. 8/16-bit ints are scaled to 0..1; float (EXR) kept as is."""
    cv2 = get_cv2()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".exr", ".dpx"):
        _check_ext_supported(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img is None:
        raise I2RotoError(f"Could not read image file: {path}")
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] >= 3:
        img = img[:, :, :3][:, :, ::-1]  # BGR(A) -> RGB
    else:
        img = np.repeat(img[:, :, :1], 3, axis=2)
    if img.dtype == np.uint8:
        out = img.astype(np.float32) * (1.0 / 255.0)
    elif img.dtype == np.uint16:
        out = img.astype(np.float32) * (1.0 / 65535.0)
    else:
        out = img.astype(np.float32)
    return np.ascontiguousarray(out)


def read_gray(path):
    """Read an image as float32 HxW (first channel / luminance-agnostic, 0..1 for ints)."""
    cv2 = get_cv2()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".exr", ".dpx"):
        _check_ext_supported(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img is None:
        raise I2RotoError(f"Could not read image file: {path}")
    if img.ndim == 3:
        img = img.max(axis=2) if img.shape[2] <= 3 else img[:, :, :3].max(axis=2)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def write_png16_gray(path, a, compression=1):
    cv2 = get_cv2()
    q = np.clip(np.rint(np.clip(a, 0.0, 1.0) * 65535.0), 0, 65535).astype(np.uint16)
    if not cv2.imwrite(path, q, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)]):
        raise I2RotoError(f"Could not write {path}")


def write_matte(path, a, fmt):
    """Write a single-channel matte. fmt: 'png16' | 'exr' | 'tiff16'."""
    cv2 = get_cv2()
    if fmt == "png16":
        write_png16_gray(path, a, compression=1)
    elif fmt == "tiff16":
        q = np.clip(np.rint(np.clip(a, 0.0, 1.0) * 65535.0), 0, 65535).astype(np.uint16)
        if not cv2.imwrite(path, q):
            raise I2RotoError(f"Could not write {path}")
    elif fmt == "exr":
        if not cv2.haveImageWriter("x.exr"):
            write_exr_half_y(path, a)  # built-in writer (OpenCV 5.x pip wheels have no OpenEXR)
            return
        params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
        if hasattr(cv2, "IMWRITE_EXR_COMPRESSION") and hasattr(cv2, "IMWRITE_EXR_COMPRESSION_ZIP"):
            params += [cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP]
        # A single-channel float image is written by OpenCV as one channel named "Y".
        if not cv2.imwrite(path, np.ascontiguousarray(a.astype(np.float32)), params):
            raise I2RotoError(f"Could not write {path}")
    else:
        raise ValueError(fmt)


def write_exr_half_y(path, a):
    """Minimal OpenEXR writer: single-part scanline, one HALF channel "Y", ZIP compression (16 lines/block).

    Used when OpenCV was built without OpenEXR. Follows the OpenEXR file layout spec (magic, version 2,
    header attributes, offset table, chunks with the ZIP predictor + byte interleave)."""
    import struct
    import zlib
    a = np.ascontiguousarray(np.clip(a, 0.0, 65504.0).astype("<f2"))
    h, w = a.shape

    def attr(name, typ, data):
        return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(data)) + data

    chlist = b"Y\0" + struct.pack("<iB3xii", 1, 0, 1, 1) + b"\0"
    box = struct.pack("<iiii", 0, 0, w - 1, h - 1)
    header = (b"\x76\x2f\x31\x01" + struct.pack("<i", 2)
              + attr("channels", "chlist", chlist)
              + attr("compression", "compression", struct.pack("<B", 3))
              + attr("dataWindow", "box2i", box)
              + attr("displayWindow", "box2i", box)
              + attr("lineOrder", "lineOrder", struct.pack("<B", 0))
              + attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
              + attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
              + attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
              + b"\0")
    nblocks = (h + 15) // 16
    chunks = []
    for b in range(nblocks):
        y0 = b * 16
        raw = a[y0:y0 + 16].tobytes()
        buf = np.frombuffer(raw, dtype=np.uint8)
        n = buf.size
        tmp = np.empty(n, dtype=np.uint8)
        half = (n + 1) // 2
        tmp[:half] = buf[0::2]
        tmp[half:] = buf[1::2]
        d = tmp.astype(np.int16)
        d[1:] = (tmp[1:].astype(np.int16) - tmp[:-1].astype(np.int16) + 128 + 256) & 0xFF
        comp = zlib.compress(d.astype(np.uint8).tobytes(), 6)
        data = comp if len(comp) < n else raw
        chunks.append(struct.pack("<ii", y0, len(data)) + data)
    offset = len(header) + 8 * nblocks
    table = []
    for c in chunks:
        table.append(struct.pack("<Q", offset))
        offset += len(c)
    with open(path, "wb") as f:
        f.write(header)
        f.write(b"".join(table))
        for c in chunks:
            f.write(c)


def write_rgba16(path, rgb, a):
    cv2 = get_cv2()
    rgba = np.concatenate([np.clip(rgb, 0, 1)[:, :, ::-1], np.clip(a, 0, 1)[:, :, None]], axis=2)
    q = np.rint(rgba * 65535.0).astype(np.uint16)
    if not cv2.imwrite(path, q, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise I2RotoError(f"Could not write {path}")


# ----------------------------------------------------------------------------- sequences

_HASH_RE = re.compile(r"#+")
_PRINTF_RE = re.compile(r"%0?(\d*)d")
_TAIL_NUM_RE = re.compile(r"^(.*?)(\d+)(\D*)$")


def _scan(directory, prefix, suffix):
    """Find files directory/prefix<digits>suffix. Returns sorted [(frame_number, path)]."""
    found = []
    try:
        names = os.listdir(directory)
    except OSError:
        return found
    pre_l, suf_l = prefix.lower(), suffix.lower()
    for name in names:
        low = name.lower()
        if not (low.startswith(pre_l) and low.endswith(suf_l)):
            continue
        mid = name[len(prefix):len(name) - len(suffix)] if suffix else name[len(prefix):]
        if mid.isdigit():
            found.append((int(mid), os.path.join(directory, name)))
    found.sort()
    return found


def find_sequence(path):
    """Resolve a sequence spec to (frames [(num, path)], base_name).

    Accepts: a folder of frames, a pattern with #### or %04d, or any one file of a numbered sequence.
    Returns None if `path` is not a sequence spec.
    """
    if os.path.isdir(path):
        groups = {}
        for name in sorted(os.listdir(path)):
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            m = _TAIL_NUM_RE.match(os.path.splitext(name)[0])
            if not m:
                continue
            key = (m.group(1), m.group(3) + os.path.splitext(name)[1])
            groups.setdefault(key, []).append(name)
        if not groups:
            raise I2RotoError(f"No numbered image frames (png/jpg/tif/exr/dpx) found in folder: {path}")
        (prefix, suffix), _ = max(groups.items(), key=lambda kv: len(kv[1]))
        frames = _scan(path, prefix, suffix)
        base = prefix.rstrip("._- ") or os.path.basename(os.path.normpath(path))
        return frames, base
    directory, name = os.path.split(path)
    directory = directory or "."
    m = _HASH_RE.search(name)
    if m:
        prefix, suffix = name[:m.start()], name[m.end():]
        return _scan(directory, prefix, suffix), prefix.rstrip("._- ") or "plate"
    m = _PRINTF_RE.search(name)
    if m:
        prefix, suffix = name[:m.start()], name[m.end():]
        return _scan(directory, prefix, suffix), prefix.rstrip("._- ") or "plate"
    ext = os.path.splitext(name)[1].lower()
    if ext in IMAGE_EXTS and os.path.isfile(path):
        m = _TAIL_NUM_RE.match(os.path.splitext(name)[0])
        if m:
            prefix, suffix = m.group(1), m.group(3) + os.path.splitext(name)[1]
            frames = _scan(directory, prefix, suffix)
            if frames:
                return frames, prefix.rstrip("._- ") or "plate"
        return [(None, path)], os.path.splitext(name)[0]
    return None


def describe_gaps(numbers, limit=20):
    gaps = []
    for a, b in zip(numbers[:-1], numbers[1:]):
        if b != a + 1:
            gaps.append((a + 1, b - 1))
    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in gaps[:limit]]
    if len(gaps) > limit:
        parts.append(f"... ({len(gaps) - limit} more)")
    return gaps, ", ".join(parts)
