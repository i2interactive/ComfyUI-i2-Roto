"""Pure geometry and scheduling: proxy window, chunk plan, detail tile grid. No ComfyUI, no IO."""
import math

WINDOWS = {
    "1920x1088 (model native, best)": (1920, 1088),
    "1536x864": (1536, 864),
    "1280x704 (low VRAM, faster)": (1280, 704),
}
WINDOW_NAMES = list(WINDOWS)
DEFAULT_WINDOW = "1920x1088 (model native, best)"

DETAIL_PROXY = "proxy only (fastest)"
DETAIL_2X = "2x proxy"
DETAIL_NATIVE = "native (max quality)"
DETAIL_MODES = [DETAIL_PROXY, DETAIL_2X, DETAIL_NATIVE]


def ceil32(x):
    return int(math.ceil(x / 32.0) * 32)


def snap_chunk_frames(n):
    """Snap to 8k+1 (nearest) and clamp to 9..145."""
    n = int(n)
    k = int(round((n - 1) / 8.0))
    return max(9, min(145, 8 * k + 1))


def proxy_geometry(w, h, ww, wh):
    s = min(ww / float(w), wh / float(h), 1.0)
    pw, ph = int(round(w * s)), int(round(h * s))
    pw, ph = max(1, min(pw, ww)), max(1, min(ph, wh))
    return {"s": s, "pw": pw, "ph": ph, "win_w": ceil32(pw), "win_h": ceil32(ph),
            "pad_w": ceil32(pw) - pw, "pad_h": ceil32(ph) - ph}


def chunk_plan(n, L):
    """Return list of chunks. Each: start, length, frames (source-relative indices, tail repeats the last real
    frame), store (list of (position_in_chunk, frame_index)) for the frames this chunk owns."""
    L = snap_chunk_frames(L)
    chunks = []
    if n <= 0:
        return chunks
    if n <= L:
        length = 8 * int(math.ceil((n - 1) / 8.0)) + 1
        frames = [min(i, n - 1) for i in range(length)]
        chunks.append({"index": 0, "start": 0, "length": length, "frames": frames,
                       "own": [(i, i) for i in range(n)]})
        return chunks
    k = int(math.ceil((n - 1) / float(L - 1)))
    for c in range(k):
        start = c * (L - 1)
        frames = [min(start + i, n - 1) for i in range(L)]
        first = 0 if c == 0 else 1
        own = [(i, start + i) for i in range(first, L) if start + i < n]
        chunks.append({"index": c, "start": start, "length": L, "frames": frames, "own": own})
    return chunks


def chunk_of_frame(chunks, f):
    """Index of the chunk that owns frame f."""
    for c in chunks:
        if c["own"] and c["own"][0][1] <= f <= c["own"][-1][1]:
            return c["index"]
    raise IndexError(f)


def working_scale(detail, s):
    if s >= 1.0:
        return 1.0
    if detail == DETAIL_NATIVE:
        return 1.0
    if detail == DETAIL_2X:
        return min(1.0, 2.0 * s)
    return s


def axis_tiles(box0, box1, extent, tile, overlap):
    """1-D tile starts covering [box0, box1) inside [0, extent), each `tile` long, >= overlap overlap,
    centred on the box, shifted inward so none leaves [0, extent)."""
    tile = min(tile, extent)
    box0, box1 = max(0, box0), min(extent, box1)
    length = box1 - box0
    if length <= tile:
        c = (box0 + box1) / 2.0
        x = int(round(c - tile / 2.0))
        return [max(0, min(extent - tile, x))]
    n = int(math.ceil((length - overlap) / float(tile - overlap)))
    n = max(n, 2)
    stride = (length - tile) / float(n - 1)
    starts = []
    for i in range(n):
        x = int(math.floor(box0 + i * stride))
        starts.append(max(0, min(extent - tile, x)))
    return starts


def tile_grid(box, work_w, work_h, win_w, win_h, overlap):
    """box = (x0, y0, x1, y1) in working-res px. Returns tiles: dict(x, y, w, h, pad_w, pad_h, win_w, win_h,
    ox, oy) where (w, h) is the content size (<= window) and (ox, oy) the effective overlap per axis."""
    tw, th = min(win_w, work_w), min(win_h, work_h)
    ox = max(0, min(overlap, tw // 2 - 1))
    oy = max(0, min(overlap, th // 2 - 1))
    xs = axis_tiles(box[0], box[2], work_w, tw, ox)
    ys = axis_tiles(box[1], box[3], work_h, th, oy)
    tiles = []
    for y in ys:
        for x in xs:
            tiles.append({"x": x, "y": y, "w": tw, "h": th, "win_w": ceil32(tw), "win_h": ceil32(th),
                          "pad_w": ceil32(tw) - tw, "pad_h": ceil32(th) - th, "ox": ox, "oy": oy})
    for i, t in enumerate(tiles):
        t["index"] = i
    return tiles
