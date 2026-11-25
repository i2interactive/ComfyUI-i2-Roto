"""Image maths for the planner and assembler: edge band, tile weights, guided upscale, edge shift, clip."""
import numpy as np

from .io_utils import get_cv2


def _disk(r):
    cv2 = get_cv2()
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def edge_band(p, r):
    """band = not (erode(fg, r) or erode(bg, r)) with fg = p > 0.98, bg = p < 0.02. Returns bool array."""
    cv2 = get_cv2()
    k = _disk(r)
    fg = (p > 0.98).astype(np.uint8)
    bg = (p < 0.02).astype(np.uint8)
    # Pixels outside the frame count as "same as the border" (no artificial band along frame edges).
    efg = cv2.erode(fg, k, borderType=cv2.BORDER_REPLICATE)
    ebg = cv2.erode(bg, k, borderType=cv2.BORDER_REPLICATE)
    return ~((efg > 0) | (ebg > 0))


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ramp_1d(n, start_ramp, end_ramp):
    """Weights along one axis: smoothstep up over `start_ramp` px, 1 inside, down over `end_ramp` px."""
    x = np.arange(n, dtype=np.float32) + 0.5
    w = np.ones(n, dtype=np.float32)
    if start_ramp > 0:
        w *= smoothstep(x / float(start_ramp))
    if end_ramp > 0:
        w *= smoothstep((n - x) / float(end_ramp))
    return w


def tile_weight(tile, work_w, work_h):
    """2-D weight for a tile (content size). No ramp on sides that sit on the frame border."""
    left = 0 if tile["x"] <= 0 else tile["ox"]
    right = 0 if tile["x"] + tile["w"] >= work_w else tile["ox"]
    top = 0 if tile["y"] <= 0 else tile["oy"]
    bottom = 0 if tile["y"] + tile["h"] >= work_h else tile["oy"]
    wx = ramp_1d(tile["w"], left, right)
    wy = ramp_1d(tile["h"], top, bottom)
    return wy[:, None] * wx[None, :]


def resize(img, w, h, interp="cubic"):
    cv2 = get_cv2()
    if img.shape[1] == w and img.shape[0] == h:
        return img
    flag = {"cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST}[interp]
    if interp == "area" and (w > img.shape[1] or h > img.shape[0]):
        flag = cv2.INTER_LINEAR
    return cv2.resize(img, (int(w), int(h)), interpolation=flag)


def gaussian(img, sigma):
    cv2 = get_cv2()
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REPLICATE)


def box(img, r):
    cv2 = get_cv2()
    return cv2.blur(img, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_coeffs(I, p, r=8, eps=1e-4):
    """He et al. guided filter, colour guide. I: HxWx3 float32, p: HxW. Returns mean_a (HxWx3), mean_b (HxW)."""
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    mI = [box(I[:, :, c], r) for c in range(3)]
    mp = box(p, r)
    cov_Ip = [box(I[:, :, c] * p, r) - mI[c] * mp for c in range(3)]
    # Covariance of I (symmetric 3x3) + eps*U
    v = {}
    for a in range(3):
        for b in range(a, 3):
            v[(a, b)] = box(I[:, :, a] * I[:, :, b], r) - mI[a] * mI[b] + (eps if a == b else 0.0)
    s00, s01, s02, s11, s12, s22 = v[(0, 0)], v[(0, 1)], v[(0, 2)], v[(1, 1)], v[(1, 2)], v[(2, 2)]
    # Inverse via adjugate.
    i00 = s11 * s22 - s12 * s12
    i01 = s02 * s12 - s01 * s22
    i02 = s01 * s12 - s02 * s11
    i11 = s00 * s22 - s02 * s02
    i12 = s01 * s02 - s00 * s12
    i22 = s00 * s11 - s01 * s01
    det = s00 * i00 + s01 * i01 + s02 * i02
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    c0, c1, c2 = cov_Ip
    a0 = (i00 * c0 + i01 * c1 + i02 * c2) / det
    a1 = (i01 * c0 + i11 * c1 + i12 * c2) / det
    a2 = (i02 * c0 + i12 * c1 + i22 * c2) / det
    b = mp - a0 * mI[0] - a1 * mI[1] - a2 * mI[2]
    ma = np.stack([box(a0, r), box(a1, r), box(a2, r)], axis=2)
    mb = box(b, r)
    return ma.astype(np.float32), mb.astype(np.float32)


def fast_guided_upscale(A_work, I_work, I_native, r=8, eps=1e-4):
    """Fast guided filter: coefficients at working res, upsampled, applied with the native guide."""
    H, W = I_native.shape[:2]
    ma, mb = guided_coeffs(I_work, A_work, r=r, eps=eps)
    ma = resize(ma, W, H, "linear")
    mb = resize(mb, W, H, "linear")
    q = (ma * I_native).sum(axis=2) + mb
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def edge_shift(a, px):
    """Choke (px < 0) or spread (px > 0) by |px| pixels with a sub-pixel blend between integer radii."""
    if abs(px) < 1e-6:
        return a
    cv2 = get_cv2()
    t = abs(float(px))
    n = int(np.floor(t))
    f = t - n
    op = cv2.dilate if px > 0 else cv2.erode

    def morph(r):
        if r <= 0:
            return a
        return op(a, _disk(r), borderType=cv2.BORDER_REPLICATE)

    lo = morph(n)
    if f < 1e-6:
        out = lo
    else:
        out = (1.0 - f) * lo + f * morph(n + 1)
    return gaussian(out.astype(np.float32), 0.5)


def clip_remap(a, black, white):
    if white <= black:
        white = black + 1e-6
    return np.clip((a - black) / (white - black), 0.0, 1.0).astype(np.float32)


def checker(h, w, size=16, dark=0.25, light=0.55):
    yy, xx = np.mgrid[0:h, 0:w]
    c = ((yy // size + xx // size) % 2).astype(np.float32)
    v = dark + (light - dark) * c
    return np.repeat(v[:, :, None], 3, axis=2)
