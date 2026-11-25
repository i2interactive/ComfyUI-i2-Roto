"""Synthetic test plate with analytically known alpha (dev/testing only).

Everything is defined in "reference pixels" of a 6144x3240 frame and scaled with the plate width, so the
same shot can be rendered at any resolution. Ground-truth alpha at a pixel is the analytic coverage
evaluated at the pixel centre; at a lower resolution the oracle is the area-average of native alpha.

Elements (all moving over time):
  body + head   : hard edge (1 px anti-aliased)
  arm           : soft, motion-blur-like falloff (~50 px)
  hair strands  : thin curves, 1-3 px wide at native
  veil          : semi-transparent region (alpha 0.5) with soft edges
"""
import json
import math
import os

import numpy as np

REF_W = 6144.0


class Synth:
    def __init__(self, width, height, frames, seed=7):
        self.W, self.H, self.N, self.seed = int(width), int(height), int(frames), int(seed)
        self.k = self.W / REF_W
        rng = np.random.RandomState(seed)
        self.strands = []
        for i in range(36):
            self.strands.append({
                "theta": -math.pi / 2 + rng.uniform(-1.25, 1.25),
                "length": rng.uniform(220, 460),
                "amp": rng.uniform(6, 22),
                "omega": rng.uniform(0.015, 0.04),
                "phase": rng.uniform(0, 2 * math.pi),
                "width": float(rng.choice([1.0, 1.5, 2.0, 3.0])),
            })

    # -------------------------------------------------------------- params / persistence
    def to_dict(self):
        return {"width": self.W, "height": self.H, "frames": self.N, "seed": self.seed}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["width"], d["height"], d["frames"], d.get("seed", 7))

    # -------------------------------------------------------------- geometry (native px)
    def _pose(self, t):
        k = self.k
        cx = (2500 + 38 * t) * k
        cy = (1950 + 25 * math.sin(t / 4.0)) * k
        return cx, cy

    def bbox(self, t):
        cx, cy = self._pose(t)
        k = self.k
        x0, x1 = cx - 1200 * k, cx + 1350 * k
        y0, y1 = cy - 1600 * k, cy + 1300 * k
        return (int(max(0, math.floor(x0))), int(max(0, math.floor(y0))),
                int(min(self.W, math.ceil(x1))), int(min(self.H, math.ceil(y1))))

    def _elements(self, t, X, Y):
        """Return list of (alpha array, rgb colour) evaluated at coordinates X, Y (native px, float32 grids of
        one rectangle). Each element is only evaluated inside its own bounding box."""
        k = self.k
        cx, cy = self._pose(t)
        gx0, gy0 = float(X[0, 0]) - 0.5, float(Y[0, 0]) - 0.5
        H, W = X.shape

        def sub(bx0, by0, bx1, by1):
            i0 = int(max(0, math.floor(bx0 - gx0))); i1 = int(min(W, math.ceil(bx1 - gx0)))
            j0 = int(max(0, math.floor(by0 - gy0))); j1 = int(min(H, math.ceil(by1 - gy0)))
            if i1 <= i0 or j1 <= j0:
                return None
            return (slice(j0, j1), slice(i0, i1))

        out = []

        def add(box_, fn, col):
            a = np.zeros_like(X)
            sl = sub(*box_)
            if sl is not None:
                a[sl] = fn(X[sl], Y[sl])
            out.append((a, col))

        m = 4 * k + 2
        # body: ellipse, hard edge
        a_, b_ = 480 * k, 880 * k
        bx, by = cx, cy + 320 * k

        def body(x, y):
            r = np.sqrt(((x - bx) / a_) ** 2 + ((y - by) / b_) ** 2)
            return np.clip(0.5 - (r - 1.0) * min(a_, b_), 0, 1)
        add((bx - a_ - m, by - b_ - m, bx + a_ + m, by + b_ + m), body, (0.80, 0.35, 0.22))
        # head: circle, hard edge
        hr = 250 * k
        hx, hy = cx, cy - 760 * k

        def head(x, y):
            return np.clip(0.5 - (np.sqrt((x - hx) ** 2 + (y - hy) ** 2) - hr), 0, 1)
        add((hx - hr - m, hy - hr - m, hx + hr + m, hy + hr + m), head, (0.85, 0.62, 0.48))
        # arm: capsule with soft (motion blur like) falloff
        p0 = (cx + 380 * k, cy - 150 * k)
        p1 = (cx + 1150 * k, cy + (-560 + 80 * math.sin(t / 3.0)) * k)
        vx_, vy_ = p1[0] - p0[0], p1[1] - p0[1]
        L2 = vx_ * vx_ + vy_ * vy_
        soft = 50 * k
        rad = 95 * k

        def arm(x, y):
            u = np.clip(((x - p0[0]) * vx_ + (y - p0[1]) * vy_) / L2, 0, 1)
            dist = np.sqrt((x - p0[0] - u * vx_) ** 2 + (y - p0[1] - u * vy_) ** 2)
            v = np.clip((rad - dist) / soft + 0.5, 0, 1)
            return v * v * (3 - 2 * v)
        e = rad + soft + m
        add((min(p0[0], p1[0]) - e, min(p0[1], p1[1]) - e, max(p0[0], p1[0]) + e, max(p0[1], p1[1]) + e),
            arm, (0.78, 0.40, 0.25))
        # veil: semi-transparent ellipse (alpha 0.5), soft 6 px edge
        va, vb = 330 * k, 420 * k
        vx, vy = cx - 760 * k, cy + 150 * k

        def veil(x, y):
            r = np.sqrt(((x - vx) / va) ** 2 + ((y - vy) / vb) ** 2)
            return 0.5 * np.clip(0.5 - (r - 1.0) * min(va, vb) / (6 * k), 0, 1)
        add((vx - va - 8 * m, vy - vb - 8 * m, vx + va + 8 * m, vy + vb + 8 * m), veil, (0.30, 0.55, 0.85))
        # hair strands from the head
        hair = np.zeros_like(X)
        for s in self.strands:
            th = s["theta"]
            ux, uy = math.cos(th), math.sin(th)        # along
            nx, ny = -uy, ux                           # across
            ox, oy = hx + ux * (hr - 8 * k), hy + uy * (hr - 8 * k)
            Ls = s["length"] * k
            amp, om, ph = s["amp"] * k, s["omega"] / max(k, 1e-6), s["phase"] + 0.25 * t
            w = max(s["width"] * k, 1e-3)
            ex, ey = ox + ux * Ls, oy + uy * Ls
            pad = amp + w + m
            sl = sub(min(ox, ex) - pad, min(oy, ey) - pad, max(ox, ex) + pad, max(oy, ey) + pad)
            if sl is None:
                continue
            x, y = X[sl], Y[sl]
            uu = (x - ox) * ux + (y - oy) * uy
            vv = (x - ox) * nx + (y - oy) * ny
            f = amp * np.sin(om * uu + ph) * np.clip(uu / (60 * k), 0, 1)
            fp = amp * om * np.cos(om * uu + ph)
            dd = np.abs(vv - f) / np.sqrt(1 + fp * fp)
            cov = np.clip(w / 2 + 0.5 - dd, 0, 1) if w >= 1 else np.clip(1 - dd, 0, 1) * w
            taper = np.clip(uu / (10 * k), 0, 1) * np.clip((Ls - uu) / (40 * k), 0, 1)
            hair[sl] = 1 - (1 - hair[sl]) * (1 - cov * taper)
        out.append((hair, (0.20, 0.12, 0.06)))
        return out

    # -------------------------------------------------------------- rendering
    def _grid(self, x0, y0, w, h):
        xs = (np.arange(x0, x0 + w, dtype=np.float32) + 0.5)
        ys = (np.arange(y0, y0 + h, dtype=np.float32) + 0.5)
        return np.meshgrid(xs, ys)

    def alpha_region(self, t, x0, y0, w, h):
        """Native-resolution ground-truth alpha for the rectangle (x0, y0, w, h). float32 HxW."""
        out = np.zeros((h, w), dtype=np.float32)
        bx0, by0, bx1, by1 = self.bbox(t)
        ix0, iy0 = max(x0, bx0), max(y0, by0)
        ix1, iy1 = min(x0 + w, bx1), min(y0 + h, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            return out
        X, Y = self._grid(ix0, iy0, ix1 - ix0, iy1 - iy0)
        acc = np.zeros_like(X)
        for a, _ in self._elements(t, X, Y):
            acc = 1 - (1 - acc) * (1 - a)
        out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = acc
        return out

    def alpha(self, t):
        return self.alpha_region(t, 0, 0, self.W, self.H)

    def background(self, t, x0=0, y0=0, w=None, h=None):
        w = self.W if w is None else w
        h = self.H if h is None else h
        k = self.k
        xs = (np.arange(x0, x0 + w, dtype=np.float32) + 0.5) / k + 30.0 * t
        ys = (np.arange(y0, y0 + h, dtype=np.float32) + 0.5) / k
        sx1, sy1 = np.sin(xs / 97.0), np.cos(ys / 131.0)
        sx2, sy2 = np.sin(xs / 23.0), np.sin(ys / 29.0)
        base = 0.38 + 0.12 * np.outer(sy1, sx1) + 0.06 * np.outer(sy2, sx2)
        img = np.empty((h, w, 3), dtype=np.float32)
        img[:, :, 0] = base * 0.9
        img[:, :, 1] = base * 1.0 + 0.04 * np.outer(np.cos(ys / 61.0), np.cos(xs / 53.0))
        img[:, :, 2] = base * 1.1
        return np.clip(img, 0, 1)

    def plate(self, t):
        """Display-space RGB plate frame (float32 HxWx3)."""
        img = self.background(t)
        bx0, by0, bx1, by1 = self.bbox(t)
        if bx1 > bx0 and by1 > by0:
            X, Y = self._grid(bx0, by0, bx1 - bx0, by1 - by0)
            region = img[by0:by1, bx0:bx1]
            for a, col in self._elements(t, X, Y):
                c = np.array(col, dtype=np.float32)
                tex = 1.0 + 0.08 * np.sin(X / (7.0 * self.k + 1e-6))[..., None]
                region[:] = region * (1 - a[..., None]) + np.clip(c * tex, 0, 1) * a[..., None]
            img[by0:by1, bx0:bx1] = region
        return img
