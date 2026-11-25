"""Generate the synthetic test plate on disk (frame by frame, bounded RAM)."""
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from i2roto.synth import Synth  # noqa: E402


def write_png_sequence(folder, width, height, frames, name="shot", start_number=1001, seed=7, bits=16):
    import cv2
    os.makedirs(folder, exist_ok=True)
    s = Synth(width, height, frames, seed)
    s.save(os.path.join(folder, "i2roto_synth.json"))
    paths = []
    for t in range(frames):
        img = s.plate(t)[:, :, ::-1]
        if bits == 16:
            q = np.rint(img * 65535).astype(np.uint16)
        else:
            q = np.rint(img * 255).astype(np.uint8)
        p = os.path.join(folder, f"{name}.{start_number + t:04d}.png")
        cv2.imwrite(p, q, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        paths.append(p)
    return s, paths


def write_exr_sequence(folder, width, height, frames, name="shot", start_number=1, seed=7):
    """Linear-light EXR plate (display plate passed through the inverse sRGB OETF)."""
    import cv2
    os.makedirs(folder, exist_ok=True)
    s = Synth(width, height, frames, seed)
    s.save(os.path.join(folder, "i2roto_synth.json"))
    for t in range(frames):
        d = s.plate(t)
        lin = np.where(d <= 0.04045, d / 12.92, ((d + 0.055) / 1.055) ** 2.4).astype(np.float32)
        p = os.path.join(folder, f"{name}.{start_number + t:04d}.exr")
        ok = cv2.imwrite(p, lin[:, :, ::-1].copy(), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        if not ok:
            raise RuntimeError("EXR write failed")
    return s


def write_video(path, width, height, frames, codec="prores", seed=7, fps=24, gop=12):
    """ProRes 422 HQ 10-bit .mov (intra) or H.264 8-bit long-GOP with B-frames (.mov/.mp4). BT.709 tagged."""
    import av
    s = Synth(width, height, frames, seed)
    s.save(os.path.join(os.path.dirname(path), "i2roto_synth.json"))
    c = av.open(path, "w")
    if codec == "prores":
        st = c.add_stream("prores_ks", rate=fps)
        st.pix_fmt = "yuv422p10le"
        st.options = {"profile": "3"}
        src_fmt, dst_fmt, scale, dt = "rgb48le", "yuv422p10le", 65535, np.uint16
    else:
        st = c.add_stream("libx264", rate=fps)
        st.pix_fmt = "yuv420p"
        st.options = {"crf": "12", "g": str(gop), "bf": "2", "preset": "ultrafast"}
        src_fmt, dst_fmt, scale, dt = "rgb24", "yuv420p", 255, np.uint8
    st.width, st.height = width, height
    st.codec_context.colorspace = 1      # BT.709
    st.codec_context.color_range = 1     # limited
    for t in range(frames):
        img = np.rint(s.plate(t) * scale).astype(dt)
        fr = av.VideoFrame.from_ndarray(img, format=src_fmt)
        fr = fr.reformat(format=dst_fmt, src_colorspace="ITU709", dst_colorspace="ITU709",
                         src_color_range="JPEG", dst_color_range="MPEG")
        fr.colorspace = 1
        fr.color_range = 1
        fr.pts = t
        fr.time_base = Fraction(1, fps)
        for pkt in st.encode(fr):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    return s


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--width", type=int, default=3072)
    ap.add_argument("--height", type=int, default=1620)
    ap.add_argument("--frames", type=int, default=41)
    a = ap.parse_args()
    write_png_sequence(a.folder, a.width, a.height, a.frames)
    print("ok", a.folder)
