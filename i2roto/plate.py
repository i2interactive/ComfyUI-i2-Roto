"""Plate access: one interface over video files (PyAV) and image sequences.

Frames are addressed by *source index* (0-based position in the clip/sequence). Raw values are returned
(float32 RGB, not colour converted); colour conversion lives in color.py.
"""
import os
from fractions import Fraction

import numpy as np

from .io_utils import (I2RotoError, clean_path, find_sequence, describe_gaps, read_image,
                       VIDEO_EXTS, IMAGE_EXTS)

_AV_COLORSPACE = {1: "ITU709", 5: "ITU601", 6: "SMPTE170M", 7: "SMPTE240M", 9: "BT2020", 10: "BT2020", 4: "FCC"}


def open_plate(path):
    p = clean_path(path)
    if not p:
        raise I2RotoError("plate_path is empty. Paste the path to a video file, a frame folder or a frame sequence.")
    ext = os.path.splitext(p)[1].lower()
    if os.path.isfile(p) and ext in VIDEO_EXTS:
        return VideoPlate(p)
    if os.path.isfile(p) and ext not in IMAGE_EXTS:
        # Unknown extension: try it as a video.
        try:
            return VideoPlate(p)
        except I2RotoError:
            raise
        except Exception:
            raise I2RotoError(f"Unsupported file type: {p}. Use a video (mov/mp4/mxf/mkv) or a png/jpg/tif/exr sequence.")
    seq = find_sequence(p)
    if seq is None:
        if not os.path.exists(os.path.dirname(p) or "."):
            raise I2RotoError(f"Folder not found: {os.path.dirname(p)}")
        raise I2RotoError(f"Plate not found: {p}")
    frames, base = seq
    if not frames:
        raise I2RotoError(f"No frames match the sequence pattern: {p}")
    return SequencePlate(frames, base, p)


class SequencePlate:
    kind = "sequence"

    def __init__(self, frames, base, spec):
        self.spec = spec
        self.base_name = base
        numbers = [n for n, _ in frames]
        if numbers[0] is None:
            self.paths = [frames[0][1]]
            self.numbers = None
        else:
            gaps, text = describe_gaps(numbers)
            if gaps:
                raise I2RotoError(
                    f"The frame sequence has gaps (missing frames: {text}). "
                    f"Fill or re-render the missing frames, or point plate_path at one continuous range."
                )
            self.paths = [p for _, p in frames]
            self.numbers = numbers
        first = read_image(self.paths[0])
        self.height, self.width = first.shape[:2]
        self.frame_count = len(self.paths)
        self.fps = None
        self.ext = os.path.splitext(self.paths[0])[1].lower()

    def frame_number(self, index):
        return None if self.numbers is None else self.numbers[index]

    def read(self, index):
        img = read_image(self.paths[index])
        if img.shape[0] != self.height or img.shape[1] != self.width:
            raise I2RotoError(f"Frame size changes inside the sequence: {self.paths[index]} is "
                              f"{img.shape[1]}x{img.shape[0]}, the first frame is {self.width}x{self.height}.")
        return img

    def iter_frames(self, indices):
        """Yield (index, frame) for the given ascending indices (repeats allowed)."""
        last_i, last_f = None, None
        for i in indices:
            if i != last_i:
                last_f = self.read(i)
                last_i = i
            yield i, last_f

    def close(self):
        pass


class VideoPlate:
    """PyAV-backed video plate. Assumes constant frame rate (frame index = round((pts - first_pts) * fps))."""
    kind = "video"
    ext = None
    numbers = None

    def __init__(self, path):
        try:
            import av  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            raise I2RotoError("PyAV (the 'av' package) is required to read video files.") from e
        self._av = av
        self.spec = path
        self.base_name = os.path.splitext(os.path.basename(path))[0]
        try:
            c = av.open(path)
        except Exception as e:
            raise I2RotoError(f"Could not open video file {path}: {e}")
        try:
            if not c.streams.video:
                raise I2RotoError(f"No video stream in {path}")
            s = c.streams.video[0]
            self.width = s.codec_context.width
            self.height = s.codec_context.height
            rate = s.average_rate or s.guessed_rate or s.base_rate
            if not rate:
                raise I2RotoError(f"Could not determine the frame rate of {path}")
            self.rate = Fraction(rate.numerator, rate.denominator)
            self.fps = float(self.rate)
            self.time_base = s.time_base
            fmt = s.codec_context.format
            depth = 8
            try:
                depth = max(c_.bits for c_ in fmt.components) if fmt is not None else 8
            except Exception:
                pass
            self.high_bit = depth > 8
            # First frame pts (edit lists / non-zero start times).
            self.first_pts = None
            for frame in c.decode(s):
                self.first_pts = frame.pts if frame.pts is not None else 0
                break
            if self.first_pts is None:
                raise I2RotoError(f"The video has no decodable frames: {path}")
            n = int(s.frames or 0)
            if n <= 0:
                c.seek(0)
                n = 0
                for pkt in c.demux(s):
                    if pkt.pts is not None and pkt.size > 0:
                        n += 1
            self.frame_count = n
        finally:
            c.close()
        self._c = None
        self._s = None
        self._next = None  # index of the next frame the open decoder will deliver (approx)
        self._gen = None

    def frame_number(self, index):
        return None

    def _index_of(self, frame):
        pts = frame.pts if frame.pts is not None else 0
        t = (pts - self.first_pts) * self.time_base
        return int(round(float(t * self.rate)))

    def _to_rgb(self, frame):
        cs = _AV_COLORSPACE.get(int(frame.colorspace or 2))
        if cs is None:
            cs = "ITU709" if self.height >= 720 else "ITU601"
        rng = "JPEG" if int(frame.color_range or 0) == 2 else "MPEG"
        fmt_name = frame.format.name
        is_rgb_src = fmt_name.startswith(("rgb", "bgr", "gbr", "argb", "abgr", "rgba", "bgra"))
        target = "rgb48le" if self.high_bit else "rgb24"
        if is_rgb_src:
            arr = frame.to_ndarray(format=target)
        else:
            arr = frame.reformat(format=target, src_colorspace=cs, dst_colorspace=cs,
                                 src_color_range=rng, dst_color_range="JPEG").to_ndarray()
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) * (1.0 / 65535.0)
        return arr.astype(np.float32) * (1.0 / 255.0)

    def _open(self):
        if self._c is None:
            self._c = self._av.open(self.spec)
            self._s = self._c.streams.video[0]
            self._s.thread_type = "AUTO"

    def _seek(self, index):
        """Seek to the keyframe at or before `index` and return a frame generator."""
        self._open()
        back = 0.0
        while True:
            t = max(0.0, index / self.fps - back)
            target = self.first_pts + int(round(t / float(self.time_base)))
            self._c.seek(max(target, 0) if index > 0 or back > 0 else self.first_pts, stream=self._s,
                         backward=True, any_frame=False)
            gen = self._c.decode(self._s)
            try:
                first = next(gen)
            except StopIteration:
                first = None
            if first is None:
                if t <= 0:
                    raise I2RotoError(f"Could not decode frame {index} of {self.spec}")
                back = back * 2 + 1.0
                continue
            if self._index_of(first) <= index or t <= 0:
                return first, gen
            back = back * 2 + 1.0  # landed after the wanted frame: seek further back

    def iter_frames(self, indices):
        """Yield (index, frame) for ascending indices. One seek, then decode forward (long-GOP safe)."""
        indices = list(indices)
        if not indices:
            return
        last = self.frame_count - 1
        wanted = [min(i, last) for i in indices]
        first, gen = self._seek(wanted[0])
        pending = first
        cache_i, cache_f = None, None
        for i in wanted:
            if i == cache_i:
                yield i, cache_f
                continue
            while True:
                if pending is None:
                    try:
                        pending = next(gen)
                    except StopIteration:
                        pending = None
                        break
                fi = self._index_of(pending)
                if fi < i:
                    pending = None
                    continue
                if fi == i:
                    cache_i, cache_f = i, self._to_rgb(pending)
                    pending = None
                break
            if cache_i != i:
                if cache_f is None:
                    raise I2RotoError(f"Could not decode frame {i} of {self.spec}")
                # Frame missing in the stream (VFR / dropped): hold the previous decoded frame.
                cache_i = i
            yield i, cache_f

    def read(self, index):
        for _, f in self.iter_frames([index]):
            return f

    def close(self):
        if self._c is not None:
            try:
                self._c.close()
            finally:
                self._c = None
                self._s = None
