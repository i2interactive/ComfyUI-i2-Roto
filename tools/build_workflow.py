#!/usr/bin/env python3
"""Generate the i2 Roto workflow JSON (UI format, frontend schema version 0.4 with subgraph definitions).

    python tools/build_workflow.py            -> workflows/i2_Roto_LTX25_AlphaGen_HiRes.json
                                                 tests/dev_stub_test.json  (CPU test graph, stub matte)

Conventions mirror Lightricks' official LTX-2.5 Alpha Gen chunked workflow: StartLoop "For" widgets
[mode, mode.start_iteration_index, mode.max_iteration, mode.step, cache_iterations] with a linked
"mode.max_iteration" widget input, EndLoop [accumulate], subgraph definitions under definitions.subgraphs
with inputNode -10 / outputNode -20, global node and link ids. Every node carries both widgets_values
(positional) and widgets_values_named.
"""
import json
import os
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

# ----------------------------------------------------------------------------- node registry
# type: (socket inputs [(name, type, optional)], widgets [(name, type)], outputs [(name, type)])
SETUP_WIDGETS = [
    ("plate_path", "STRING"), ("start_frame", "INT"), ("frame_count", "INT"), ("plate_color", "COMBO"),
    ("window", "COMBO"), ("chunk_frames", "INT"), ("detail", "COMBO"), ("tile_overlap", "INT"), ("band_px", "INT"),
    ("tile_trust", "FLOAT"), ("pin_chunks", "BOOLEAN"), ("black_clip", "FLOAT"), ("white_clip", "FLOAT"),
    ("invert", "BOOLEAN"), ("edge_shift_px", "FLOAT"), ("guided_upscale", "BOOLEAN"), ("garbage_matte_path", "STRING"),
    ("output_folder", "STRING"), ("output_format", "COMBO"), ("write_rgba", "BOOLEAN"), ("start_number", "INT"),
    ("keep_cache", "BOOLEAN"),
]
REG = {
    "I2RotoSetup": ([], SETUP_WIDGETS, [("run", "I2ROTO_RUN"), ("proxy_jobs", "INT"), ("blank_pin", "IMAGE"),
                                        ("first_frame", "IMAGE"), ("info", "STRING")]),
    "I2RotoLoadJob": ([("run", "I2ROTO_RUN", False), ("job_index", "INT", False)], [("stage", "COMBO")],
                      [("images", "IMAGE"), ("width", "INT"), ("height", "INT"), ("length", "INT"),
                       ("is_first_chunk", "BOOLEAN"), ("job", "I2ROTO_JOB")]),
    "I2RotoStoreJob": ([("job", "I2ROTO_JOB", False), ("matte", "IMAGE", False)], [],
                       [("next_pin", "IMAGE"), ("done", "I2ROTO_RUN")]),
    "I2RotoPlanDetail": ([("run", "I2ROTO_RUN", False), ("proxy_done", "*", False)], [],
                         [("run", "I2ROTO_RUN"), ("detail_jobs", "INT"), ("info", "STRING")]),
    "I2RotoAssemble": ([("run", "I2ROTO_RUN", False), ("detail_done", "*", False)], [],
                       [("matte_preview", "IMAGE"), ("comp_preview", "IMAGE"), ("output_folder", "STRING"),
                        ("report", "STRING")]),
    "I2RotoStubMatte": ([("images", "IMAGE", False), ("job", "I2ROTO_JOB", False)], [], [("matte", "IMAGE")]),
    "PreviewImage": ([("images", "IMAGE", False)], [], []),
    "PreviewAny": ([("source", "*", False)], [], [("STRING", "STRING")]),
    "MarkdownNote": ([], [("text", "STRING")], []),
    "StartLoop": ([("parent_iteration", "INT", True), ("initial_iteration_value", "*", True)],
                  [("mode", "COMBO"), ("mode.start_iteration_index", "INT"), ("mode.max_iteration", "INT"),
                   ("mode.step", "INT"), ("cache_iterations", "BOOLEAN")],
                  [("iteration_index", "INT"), ("is_first", "BOOLEAN"), ("is_last", "BOOLEAN"), ("list_item", "*"),
                   ("current_iteration_value", "IMAGE")]),
    "EndLoop": ([("output_value", "*", True), ("next_iteration_value", "*", True),
                 ("terminations.termination0", "*", True)], [("accumulate", "BOOLEAN")],
                [("outputs", "I2ROTO_RUN")]),
    "LTXVPreprocess": ([("image", "IMAGE", False)], [("img_compression", "INT")], [("output_image", "IMAGE")]),
    "UNETLoader": ([], [("unet_name", "COMBO"), ("weight_dtype", "COMBO")], [("MODEL", "MODEL")]),
    "LoraLoaderModelOnly": ([("model", "MODEL", False)], [("lora_name", "COMBO"), ("strength_model", "FLOAT")],
                            [("MODEL", "MODEL")]),
    "GetICLoRAParameters": ([("iclora_model", "MODEL", False)], [], [("iclora_parameters", "IC_LORA_PARAMETERS")]),
    "VAELoader": ([], [("vae_name", "COMBO")], [("VAE", "VAE")]),
    "CLIPLoader": ([], [("clip_name", "COMBO"), ("type", "COMBO"), ("device", "COMBO")], [("CLIP", "CLIP")]),
    "CLIPTextEncode": ([("clip", "CLIP", False)], [("text", "STRING")], [("CONDITIONING", "CONDITIONING")]),
    "ConditioningZeroOut": ([("conditioning", "CONDITIONING", False)], [], [("CONDITIONING", "CONDITIONING")]),
    "LTXVConditioning": ([("positive", "CONDITIONING", False), ("negative", "CONDITIONING", False)],
                         [("frame_rate", "FLOAT")], [("positive", "CONDITIONING"), ("negative", "CONDITIONING")]),
    "KSamplerSelect": ([], [("sampler_name", "COMBO")], [("SAMPLER", "SAMPLER")]),
    "ManualSigmas": ([], [("sigmas", "STRING")], [("SIGMAS", "SIGMAS")]),
    "RandomNoise": ([], [("noise_seed", "INT"), ("control_after_generate", "COMBO")], [("NOISE", "NOISE")]),
    "EmptyLTXVLatentVideo": ([], [("width", "INT"), ("height", "INT"), ("length", "INT"), ("batch_size", "INT")],
                             [("LATENT", "LATENT")]),
    "LTXVAddGuide": ([("positive", "CONDITIONING", False), ("negative", "CONDITIONING", False), ("vae", "VAE", False),
                      ("latent", "LATENT", False), ("image", "IMAGE", False), ("attention_mask", "MASK", True),
                      ("iclora_parameters", "IC_LORA_PARAMETERS", True)],
                     [("frame_idx", "INT"), ("strength", "FLOAT")],
                     [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")]),
    "LTXVImgToVideoInplace": ([("vae", "VAE", False), ("image", "IMAGE", False), ("latent", "LATENT", False)],
                              [("strength", "FLOAT"), ("bypass", "BOOLEAN")], [("latent", "LATENT")]),
    "LTXVEmptyLatentAudio": ([("audio_vae", "VAE", False)],
                             [("frames_number", "INT"), ("frame_rate", "FLOAT,INT"), ("batch_size", "INT")],
                             [("Latent", "LATENT")]),
    "LTXVConcatAVLatent": ([("video_latent", "LATENT", False), ("audio_latent", "LATENT", False)], [],
                           [("latent", "LATENT")]),
    "LTXVDualCFGGuider": ([("model", "MODEL", False), ("positive", "CONDITIONING", False),
                           ("negative", "CONDITIONING", False)], [("video_cfg", "FLOAT"), ("audio_cfg", "FLOAT")],
                          [("GUIDER", "GUIDER")]),
    "SamplerCustomAdvanced": ([("noise", "NOISE", False), ("guider", "GUIDER", False), ("sampler", "SAMPLER", False),
                               ("sigmas", "SIGMAS", False), ("latent_image", "LATENT", False)], [],
                              [("output", "LATENT"), ("denoised_output", "LATENT")]),
    "LTXVSeparateAVLatent": ([("av_latent", "LATENT", False)], [],
                             [("video_latent", "LATENT"), ("audio_latent", "LATENT")]),
    "LTXVCropGuides": ([("positive", "CONDITIONING", False), ("negative", "CONDITIONING", False),
                        ("latent", "LATENT", False)], [],
                       [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")]),
    "VAEDecodeTiled": ([("samples", "LATENT", False), ("vae", "VAE", False)],
                       [("tile_size", "INT"), ("overlap", "INT"), ("temporal_size", "INT"),
                        ("temporal_overlap", "INT")], [("IMAGE", "IMAGE")]),
    "CreateVideo": ([("images", "IMAGE", False), ("audio", "AUDIO", True)],
                    [("fps", "FLOAT"), ("bit_depth", "COMBO"), ("color_space", "COMBO"),
                     ("chroma_subsampling", "COMBO"), ("codec", "COMBO")], [("VIDEO", "VIDEO")]),
    "SaveVideo": ([("video", "VIDEO", False)],
                  [("filename_prefix", "STRING"), ("format", "COMBO"), ("format.codec", "COMBO"), ("codec", "COMBO")],
                  [("video", "VIDEO")]),
}

HF = "https://huggingface.co/Lightricks"
MODELS = {
    "unet": ("ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "diffusion_models",
             f"{HF}/LTX-2.5/resolve/main/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
    # Shipped value is the target machine's subfolder path (Windows separator): models/loras/LTX/...
    "lora": ("LTX\\ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors", "loras",
             f"{HF}/LTX-2.5-22b-IC-LoRA-Alpha-Gen/resolve/main/ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"),
    "vvae": ("ltx-2.5-video-vae-bf16.safetensors", "vae",
             f"{HF}/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors"),
    "avae": ("ltx-2.5-audio-vae-bf16.safetensors", "vae",
             f"{HF}/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors"),
    "clip": ("gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "text_encoders",
             f"{HF}/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
}
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

GROUP_COLORS = {"1": "#3f789e", "2": "#8a5a44", "3": "#4a7a4a", "4": "#6b5b95", "5": "#8a7a3a"}
NODE_COLORS = {"setup": ("#223", "#335"), "i2": ("#232", "#353"), "loop": ("#432", "#653")}


def _uuid(seed):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "i2roto/" + seed))


class Ids:
    def __init__(self):
        self.node = 0
        self.link = 0

    def n(self):
        self.node += 1
        return self.node

    def l(self):  # noqa: E743
        self.link += 1
        return self.link


class Node:
    def __init__(self, ids, type_, pos, size=(320, 120), title=None, widgets=None, props=None, colors=None):
        self.id = ids.n()
        self.type = type_
        self.pos = list(pos)
        self.size = list(size)
        self.title = title
        reg = REG.get(type_)
        self.sockets, self.widget_defs, self.out_defs = reg if reg else ([], [], [])
        self.widgets = dict(widgets or {})
        unknown = set(self.widgets) - {w for w, _ in self.widget_defs}
        if unknown:
            raise ValueError(f"{type_}: unknown widgets {unknown}")
        self.in_links = {}            # input name -> link id
        self.out_links = {}           # output index -> [link ids]
        self.props = dict(props or {})
        self.colors = colors
        self.order = 0
        self.out_type_override = {}

    def input_list(self):
        """Socket inputs (all), then linked widget inputs (widget order). Returns [(name, type, kind)]."""
        res = [(n, t, "socket_opt" if opt else "socket") for n, t, opt in self.sockets]
        for n, t in self.widget_defs:
            if n in self.in_links:
                res.append((n, t, "widget"))
        return res

    def slot(self, name):
        for i, (n, _, _) in enumerate(self.input_list()):
            if n == name:
                return i
        raise KeyError(f"{self.type}: no input {name}")

    def out_index(self, name):
        for i, (n, _) in enumerate(self.out_defs):
            if n == name:
                return i
        raise KeyError(f"{self.type}: no output {name}")

    def to_json(self, links_by_id):
        inputs = []
        for n, t, kind in self.input_list():
            d = {"localized_name": n.split(".")[-1], "name": n, "type": t}
            if kind == "socket_opt":
                d["shape"] = 7
            if kind == "widget":
                d["widget"] = {"name": n}
            if n.startswith("terminations."):
                d["label"] = n.split(".")[-1]
            d["link"] = self.in_links.get(n)
            inputs.append(d)
        outputs = []
        for i, (n, t) in enumerate(self.out_defs):
            t = self.out_type_override.get(n, t)
            d = {"localized_name": n, "name": n, "type": t, "links": list(self.out_links.get(i, []))}
            if self.type == "EndLoop":
                d["shape"] = 6
            outputs.append(d)
        wv = [self.widgets[n] for n, _ in self.widget_defs if n in self.widgets]
        named = {n: self.widgets[n] for n, _ in self.widget_defs if n in self.widgets}
        props = {"Node name for S&R": self.type}
        props.update(self.props)
        j = {"id": self.id, "type": self.type, "pos": self.pos, "size": self.size, "flags": {}, "order": self.order,
             "mode": 0, "inputs": inputs, "outputs": outputs}
        if self.title:
            j["title"] = self.title
        j["properties"] = props
        j["widgets_values"] = wv
        if named:
            j["widgets_values_named"] = named
        if self.colors:
            j["color"], j["bgcolor"] = self.colors
        return j


class SubgraphInstance(Node):
    def __init__(self, ids, sg, pos, size=(340, 360), title=None):
        self.sg = sg
        super().__init__(ids, sg.id, pos, size, title)
        self.sockets = [(n, t, False) for n, t in sg.inputs]
        self.widget_defs = []
        self.out_defs = list(sg.outputs)

    def to_json(self, links_by_id):
        j = super().to_json(links_by_id)
        for i in j["inputs"]:
            i.pop("localized_name", None)
        for o in j["outputs"]:
            o.pop("localized_name", None)
        j["properties"] = {}
        j.pop("widgets_values_named", None)
        return j


class Graph:
    """A (sub)graph: nodes + links. Links resolved to slot indices at serialisation time."""

    def __init__(self, ids):
        self.ids = ids
        self.nodes = []
        self.links = []  # (id, src, src_slot, dst, dst_name, type)

    def add(self, node):
        self.nodes.append(node)
        return node

    def link(self, src, out, dst, inp, type_=None):
        """src/dst: Node or ('IN', index) / ('OUT', index) for subgraph boundary."""
        lid = self.ids.l()
        if isinstance(src, Node):
            si = src.out_index(out) if isinstance(out, str) else out
            t = type_ or src.out_type_override.get(src.out_defs[si][0], src.out_defs[si][1])
            src.out_links.setdefault(si, []).append(lid)
        else:
            si = src[1]
            t = type_
        if isinstance(dst, Node):
            if inp in dst.in_links:
                raise ValueError(f"{dst.type}.{inp} linked twice")
            dst.in_links[inp] = lid
        self.links.append([lid, src, si, dst, inp, t])
        return lid

    def compute_order(self):
        # Topological order (for the 'order' field only; the frontend recomputes it).
        deps = {n.id: set() for n in self.nodes}
        for _, s, _, d, _, _ in self.links:
            if isinstance(s, Node) and isinstance(d, Node):
                deps[d.id].add(s.id)
        done, order = set(), 0
        byid = {n.id: n for n in self.nodes}
        while len(done) < len(self.nodes):
            ready = [i for i in deps if i not in done and deps[i] <= done]
            if not ready:
                ready = [i for i in deps if i not in done]
            for i in sorted(ready):
                byid[i].order = order
                order += 1
                done.add(i)

    def resolved_links(self):
        out = []
        for lid, s, si, d, inp, t in self.links:
            sid = s.id if isinstance(s, Node) else -10
            if isinstance(d, Node):
                did, dslot = d.id, d.slot(inp)
            else:
                did, dslot = -20, d[1]
            out.append((lid, sid, si, did, dslot, t))
        return out


class Subgraph(Graph):
    def __init__(self, ids, name, inputs, outputs):
        super().__init__(ids)
        self.name = name
        self.id = _uuid("subgraph/" + name)
        self.inputs = inputs      # [(name, type)]
        self.outputs = outputs    # [(name, type)]

    def IN(self, name):
        return ("IN", [n for n, _ in self.inputs].index(name))

    def OUT(self, name):
        return ("OUT", [n for n, _ in self.outputs].index(name))

    def link_in(self, name, dst, inp):
        t = dict(self.inputs)[name]
        return self.link(self.IN(name), None, dst, inp, type_=t)

    def link_out(self, src, out, name):
        t = dict(self.outputs)[name]
        return self.link(src, out, self.OUT(name), None, type_=t)

    def to_json(self, state):
        self.compute_order()
        links = self.resolved_links()
        in_ids = {i: [] for i in range(len(self.inputs))}
        out_ids = {i: [] for i in range(len(self.outputs))}
        for lid, sid, si, did, dslot, t in links:
            if sid == -10:
                in_ids[si].append(lid)
            if did == -20:
                out_ids[dslot].append(lid)
        xs = [n.pos[0] for n in self.nodes]
        ys = [n.pos[1] for n in self.nodes]
        maxx = max(n.pos[0] + n.size[0] for n in self.nodes)
        return {
            "id": self.id, "version": 1, "state": state, "revision": 0, "config": {}, "name": self.name,
            "inputNode": {"id": -10, "bounding": [min(xs) - 300, min(ys), 160, 40 + 20 * len(self.inputs)]},
            "outputNode": {"id": -20, "bounding": [maxx + 140, min(ys), 160, 60]},
            "inputs": [{"id": _uuid(f"{self.name}/in/{n}"), "name": n, "type": t, "linkIds": in_ids[i],
                        "localized_name": n, "pos": [0, 0]} for i, (n, t) in enumerate(self.inputs)],
            "outputs": [{"id": _uuid(f"{self.name}/out/{n}"), "name": n, "type": t, "linkIds": out_ids[i],
                         "localized_name": n, "pos": [0, 0]} for i, (n, t) in enumerate(self.outputs)],
            "widgets": [],
            "nodes": [n.to_json(None) for n in self.nodes],
            "groups": [],
            "links": [{"id": lid, "origin_id": sid, "origin_slot": si, "target_id": did, "target_slot": dslot,
                       "type": t} for lid, sid, si, did, dslot, t in links],
            "extra": {},
        }


# ----------------------------------------------------------------------------- subgraphs

MATTE_INPUTS = [("images", "IMAGE"), ("width", "INT"), ("height", "INT"), ("length", "INT"), ("pin", "*"),
                ("is_first_chunk", "BOOLEAN"), ("model", "MODEL"), ("positive", "CONDITIONING"),
                ("negative", "CONDITIONING"), ("iclora_parameters", "IC_LORA_PARAMETERS"), ("video_vae", "VAE"),
                ("audio_vae", "VAE"), ("noise", "NOISE"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS")]


def build_matte_subgraph(ids):
    sg = Subgraph(ids, "Matte one job (LTX Alpha Gen)", MATTE_INPUTS, [("matte", "IMAGE")])
    A = sg.add
    lat = A(Node(ids, "EmptyLTXVLatentVideo", (0, 0), (300, 160), "Empty latent (job size)",
                 {"width": 1280, "height": 704, "length": 49, "batch_size": 1}))
    guide = A(Node(ids, "LTXVAddGuide", (360, 0), (300, 200), "IC-LoRA reference = job frames",
                   {"frame_idx": 0, "strength": 1}))
    pin = A(Node(ids, "LTXVImgToVideoInplace", (720, 0), (300, 140), "Pin frame 0 (bypassed on first chunk)",
                 {"strength": 1, "bypass": False}))
    aud = A(Node(ids, "LTXVEmptyLatentAudio", (360, 260), (300, 140), "Empty audio latent",
                 {"frames_number": 49, "frame_rate": 24, "batch_size": 1}))
    cat = A(Node(ids, "LTXVConcatAVLatent", (1080, 0), (260, 80)))
    gd = A(Node(ids, "LTXVDualCFGGuider", (720, 200), (300, 140), None, {"video_cfg": 1, "audio_cfg": 1}))
    smp = A(Node(ids, "SamplerCustomAdvanced", (1400, 0), (280, 140)))
    sep = A(Node(ids, "LTXVSeparateAVLatent", (1740, 0), (260, 80)))
    crop = A(Node(ids, "LTXVCropGuides", (2060, 0), (260, 100), "Remove reference frames"))
    dec = A(Node(ids, "VAEDecodeTiled", (2380, 0), (300, 160), "Decode matte",
                 {"tile_size": 512, "overlap": 64, "temporal_size": 128, "temporal_overlap": 32}))
    L = sg.link
    sg.link_in("width", lat, "width")
    sg.link_in("height", lat, "height")
    sg.link_in("length", lat, "length")
    sg.link_in("positive", guide, "positive")
    sg.link_in("negative", guide, "negative")
    sg.link_in("video_vae", guide, "vae")
    L(lat, "LATENT", guide, "latent")
    sg.link_in("images", guide, "image")
    sg.link_in("iclora_parameters", guide, "iclora_parameters")
    sg.link_in("video_vae", pin, "vae")
    sg.link_in("pin", pin, "image")
    L(guide, "latent", pin, "latent")
    sg.link_in("is_first_chunk", pin, "bypass")
    sg.link_in("audio_vae", aud, "audio_vae")
    sg.link_in("length", aud, "frames_number")
    L(pin, "latent", cat, "video_latent")
    L(aud, "Latent", cat, "audio_latent")
    sg.link_in("model", gd, "model")
    L(guide, "positive", gd, "positive")
    L(guide, "negative", gd, "negative")
    sg.link_in("noise", smp, "noise")
    L(gd, "GUIDER", smp, "guider")
    sg.link_in("sampler", smp, "sampler")
    sg.link_in("sigmas", smp, "sigmas")
    L(cat, "latent", smp, "latent_image")
    L(smp, "denoised_output", sep, "av_latent")
    L(guide, "positive", crop, "positive")
    L(guide, "negative", crop, "negative")
    L(sep, "video_latent", crop, "latent")
    L(crop, "latent", dec, "samples")
    sg.link_in("video_vae", dec, "vae")
    sg.link_out(dec, "IMAGE", "matte")
    return sg


STUB_INPUTS = [("images", "IMAGE"), ("job", "I2ROTO_JOB")]


def build_stub_subgraph(ids):
    sg = Subgraph(ids, "Matte one job (STUB, dev test only)", STUB_INPUTS, [("matte", "IMAGE")])
    st = sg.add(Node(ids, "I2RotoStubMatte", (0, 0), (300, 80), "Stub matte (ground truth, no model)"))
    sg.link_in("images", st, "images")
    sg.link_in("job", st, "job")
    sg.link_out(st, "matte", "matte")
    return sg


# ----------------------------------------------------------------------------- notes

NOTES = {
    "1": ("## 1 · Plate + settings\n\nPaste the plate path into **i2 Roto · Setup** (video file, frame folder, or "
          "`shot.####.exr`). Every setting lives on this node.\n\n- **window**: 1280x704 for 8 GB cards, 1920x1088 "
          "is the model's native size.\n- **detail**: `native` = sharp native-res edges (slowest), `proxy only` = "
          "one pass.\n- Check the info box and the first-frame preview before a long run."),
    "2": ("## 2 · Models\n\nFive files:\n\n- `models/diffusion_models/` ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors\n"
          "- `models/loras/LTX/` ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors (re-pick it in the LoRA node if yours sits elsewhere)\n"
          "- `models/vae/` ltx-2.5-video-vae-bf16.safetensors\n"
          "- `models/vae/` ltx-2.5-audio-vae-bf16.safetensors\n"
          "- `models/text_encoders/` gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors\n\n"
          "The Alpha Gen LoRA is gated: accept the license at "
          "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen first.\n\n"
          "No prompt is used. Frame rate conditioning is fixed at 24."),
    "3": ("## 3 · Proxy pass\n\nThe whole frame, scaled to fit the window, one chunk per loop iteration. "
          "Chunks share one frame; the last matte frame of a chunk is pinned as frame 0 of the next.\n\n"
          "Mattes go to a cache on disk, not into RAM."),
    "4": ("## 4 · Detail pass\n\nPlans window-size tiles along the matte's edge band at native (or 2x proxy) "
          "resolution and runs only the tile x chunk jobs the edge actually touches.\n\nZero jobs is normal for "
          "`proxy only` or plates that already fit the window."),
    "5": ("## 5 · Output\n\nBlends the tiles into the upscaled proxy inside the edge band and writes the matte "
          "sequence at the plate's native resolution, one frame at a time.\n\nThe two videos are small QC previews "
          "(matte, and plate over a checker)."),
}


# ----------------------------------------------------------------------------- main graph

def build(stub=False):
    ids = Ids()
    g = Graph(ids)
    sg = build_stub_subgraph(ids) if stub else build_matte_subgraph(ids)
    A = g.add
    groups = []

    def group(key, title, nodes, pad=40):
        x0 = min(n.pos[0] for n in nodes) - pad
        y0 = min(n.pos[1] for n in nodes) - pad - 40
        x1 = max(n.pos[0] + n.size[0] for n in nodes) + pad
        y1 = max(n.pos[1] + n.size[1] for n in nodes) + pad
        groups.append({"id": len(groups) + 1, "title": title, "bounding": [x0, y0, x1 - x0, y1 - y0],
                       "color": GROUP_COLORS[key], "font_size": 24, "flags": {}})

    def note(key, pos, size=(400, 260)):
        return A(Node(ids, "MarkdownNote", pos, size, "Help", {"text": NOTES[key]}))

    # 1 · Plate + settings -------------------------------------------------------------
    setup_w = {
        "plate_path": "", "start_frame": 0, "frame_count": 0, "plate_color": "rec709 / sRGB (use as is)",
        "window": "1920x1088 (model native, best)", "chunk_frames": 49, "detail": "native (max quality)",
        "tile_overlap": 256, "band_px": 48, "tile_trust": 1.0, "pin_chunks": True, "black_clip": 0.005,
        "white_clip": 0.995, "invert": False, "edge_shift_px": 0.0, "guided_upscale": True, "garbage_matte_path": "",
        "output_folder": "", "output_format": "PNG 16-bit", "write_rgba": False, "start_number": 1001,
        "keep_cache": False,
    }
    setup = A(Node(ids, "I2RotoSetup", (0, 0), (420, 700), "i2 Roto · Setup", setup_w, colors=NODE_COLORS["setup"]))
    n1 = note("1", (0, 760), (420, 300))
    prev_first = A(Node(ids, "PreviewImage", (480, 0), (420, 340), "First frame (display space)"))
    info1 = A(Node(ids, "PreviewAny", (480, 400), (420, 300), "Setup info"))
    g.link(setup, "first_frame", prev_first, "images")
    g.link(setup, "info", info1, "source")
    group("1", "1 · Plate + settings", [setup, n1, prev_first, info1])

    # 2 · Models -----------------------------------------------------------------------
    X2 = 1000
    model = clip_pos = clip_neg = iclora = vvae = avae = noise = sampler = sigmas = None
    model_nodes = []
    if not stub:
        def mprops(key):
            name, folder, url = MODELS[key]
            return {"models": [{"name": name, "url": url, "directory": folder}]}
        unet = A(Node(ids, "UNETLoader", (X2, 0), (400, 110), "Load transformer (distilled, int8)",
                      {"unet_name": MODELS["unet"][0], "weight_dtype": "default"}, props=mprops("unet")))
        lora = A(Node(ids, "LoraLoaderModelOnly", (X2, 160), (400, 110), "Alpha Gen IC-LoRA",
                      {"lora_name": MODELS["lora"][0], "strength_model": 1.0}, props=mprops("lora")))
        icp = A(Node(ids, "GetICLoRAParameters", (X2, 320), (400, 60), "IC-LoRA parameters"))
        vv = A(Node(ids, "VAELoader", (X2, 430), (400, 80), "Video VAE", {"vae_name": MODELS["vvae"][0]},
                    props=mprops("vvae")))
        av = A(Node(ids, "VAELoader", (X2, 560), (400, 80), "Audio VAE", {"vae_name": MODELS["avae"][0]},
                    props=mprops("avae")))
        cl = A(Node(ids, "CLIPLoader", (X2, 690), (400, 130), "Text encoder (Gemma, int8)",
                    {"clip_name": MODELS["clip"][0], "type": "ltxv", "device": "default"}, props=mprops("clip")))
        te = A(Node(ids, "CLIPTextEncode", (X2 + 460, 0), (380, 140), "Empty prompt", {"text": ""}))
        zo = A(Node(ids, "ConditioningZeroOut", (X2 + 460, 190), (380, 60), "Negative = zero"))
        cond = A(Node(ids, "LTXVConditioning", (X2 + 460, 300), (380, 100), "Frame rate 24 (fixed)",
                      {"frame_rate": 24}))
        ks = A(Node(ids, "KSamplerSelect", (X2 + 460, 450), (380, 60), None, {"sampler_name": "euler"}))
        ms = A(Node(ids, "ManualSigmas", (X2 + 460, 560), (380, 80), "Distilled sigmas", {"sigmas": SIGMAS}))
        rn = A(Node(ids, "RandomNoise", (X2 + 460, 690), (380, 100), None,
                    {"noise_seed": 1234, "control_after_generate": "fixed"}))
        g.link(unet, "MODEL", lora, "model")
        g.link(lora, "MODEL", icp, "iclora_model")
        g.link(cl, "CLIP", te, "clip")
        g.link(te, "CONDITIONING", zo, "conditioning")
        g.link(te, "CONDITIONING", cond, "positive")
        g.link(zo, "CONDITIONING", cond, "negative")
        model, iclora, vvae, avae = (lora, "MODEL"), (icp, "iclora_parameters"), (vv, "VAE"), (av, "VAE")
        clip_pos, clip_neg = (cond, "positive"), (cond, "negative")
        noise, sampler, sigmas = (rn, "NOISE"), (ks, "SAMPLER"), (ms, "SIGMAS")
        model_nodes = [unet, lora, icp, vv, av, cl, te, zo, cond, ks, ms, rn]
        n2 = note("2", (X2, 860), (840, 300))
        group("2", "2 · Models", model_nodes + [n2])
        X3 = X2 + 940
    else:
        X3 = X2

    # 3/4 · loops ----------------------------------------------------------------------
    def loop(x, y, stage, run_src, max_src, init_src, title):
        start = A(Node(ids, "StartLoop", (x, y), (300, 220), f"Loop start: {stage} jobs",
                       {"mode": "For", "mode.start_iteration_index": 0, "mode.max_iteration": 1, "mode.step": 1,
                        "cache_iterations": False}, colors=NODE_COLORS["loop"]))
        load = A(Node(ids, "I2RotoLoadJob", (x + 360, y), (300, 180), f"i2 Roto · Load Job ({stage})",
                      {"stage": stage}, colors=NODE_COLORS["i2"]))
        inst = A(SubgraphInstance(ids, sg, (x + 720, y), (320, 380), f"Matte one job ({stage})"))
        store = A(Node(ids, "I2RotoStoreJob", (x + 1100, y), (280, 100), f"i2 Roto · Store Job ({stage})",
                       colors=NODE_COLORS["i2"]))
        pre = A(Node(ids, "LTXVPreprocess", (x + 1100, y + 160), (280, 80), "Next pin (CRF 18)",
                     {"img_compression": 18}))
        end = A(Node(ids, "EndLoop", (x + 1440, y), (280, 140), f"Loop end: {stage}",
                     {"accumulate": False}, colors=NODE_COLORS["loop"]))
        g.link(init_src[0], init_src[1], start, "initial_iteration_value")
        g.link(max_src[0], max_src[1], start, "mode.max_iteration")
        g.link(run_src[0], run_src[1], load, "run")
        g.link(start, "iteration_index", load, "job_index")
        if stub:
            g.link(load, "images", inst, "images")
            g.link(load, "job", inst, "job")
        else:
            g.link(load, "images", inst, "images")
            g.link(load, "width", inst, "width")
            g.link(load, "height", inst, "height")
            g.link(load, "length", inst, "length")
            g.link(start, "current_iteration_value", inst, "pin", type_="*")
            g.link(load, "is_first_chunk", inst, "is_first_chunk")
            g.link(model[0], model[1], inst, "model")
            g.link(clip_pos[0], clip_pos[1], inst, "positive")
            g.link(clip_neg[0], clip_neg[1], inst, "negative")
            g.link(iclora[0], iclora[1], inst, "iclora_parameters")
            g.link(vvae[0], vvae[1], inst, "video_vae")
            g.link(avae[0], avae[1], inst, "audio_vae")
            g.link(noise[0], noise[1], inst, "noise")
            g.link(sampler[0], sampler[1], inst, "sampler")
            g.link(sigmas[0], sigmas[1], inst, "sigmas")
        g.link(load, "job", store, "job")
        g.link(inst, "matte", store, "matte")
        g.link(store, "next_pin", pre, "image")
        g.link(store, "done", end, "output_value", type_="*")
        g.link(pre, "output_image", end, "next_iteration_value", type_="*")
        if stub:
            # In the stub graph the pin is not consumed by a model; route it through a terminal-free path is
            # not allowed (it would escape the loop), so it is simply unused. Nothing to link.
            pass
        return start, load, inst, store, pre, end

    a = loop(X3, 0, "proxy", (setup, "run"), (setup, "proxy_jobs"), (setup, "blank_pin"), "proxy")
    n3 = note("3", (X3, 440), (700, 220))
    group("3", "3 · Proxy pass", list(a) + [n3])

    Y4 = 800
    plan = A(Node(ids, "I2RotoPlanDetail", (X3, Y4), (300, 100), "i2 Roto · Plan Detail Pass",
                  colors=NODE_COLORS["i2"]))
    g.link(setup, "run", plan, "run")
    g.link(a[5], "outputs", plan, "proxy_done", type_="*")
    plan_info = A(Node(ids, "PreviewAny", (X3, Y4 + 160), (300, 240), "Detail plan"))
    g.link(plan, "info", plan_info, "source")
    b = loop(X3 + 360, Y4, "detail", (plan, "run"), (plan, "detail_jobs"), (setup, "blank_pin"), "detail")
    n4 = note("4", (X3 + 360, Y4 + 440), (700, 220))
    group("4", "4 · Detail pass", [plan, plan_info] + list(b) + [n4])

    # 5 · Output -----------------------------------------------------------------------
    X5 = X3 + 2200
    asm = A(Node(ids, "I2RotoAssemble", (X5, 0), (320, 140), "i2 Roto · Assemble + Write",
                 colors=NODE_COLORS["setup"]))
    g.link(plan, "run", asm, "run")
    g.link(b[5], "outputs", asm, "detail_done", type_="*")
    vids = []
    for i, (out, prefix) in enumerate([("matte_preview", "i2roto/qc_matte"), ("comp_preview", "i2roto/qc_comp")]):
        cv = A(Node(ids, "CreateVideo", (X5 + 380, i * 420), (300, 180), f"QC video: {out}",
                    {"fps": 24, "bit_depth": "auto", "color_space": "sRGB", "chroma_subsampling": "auto",
                     "codec": "none"}))
        sv = A(Node(ids, "SaveVideo", (X5 + 740, i * 420), (360, 380), None,
                    {"filename_prefix": prefix, "format": "auto", "format.codec": "auto", "codec": "auto"}))
        g.link(asm, out, cv, "images")
        g.link(cv, "VIDEO", sv, "video")
        vids += [cv, sv]
    rep = A(Node(ids, "PreviewAny", (X5, 200), (320, 300), "Report"))
    fol = A(Node(ids, "PreviewAny", (X5, 560), (320, 120), "Output folder"))
    g.link(asm, "report", rep, "source")
    g.link(asm, "output_folder", fol, "source")
    n5 = note("5", (X5, 740), (680, 200))
    group("5", "5 · Output", [asm, rep, fol, n5] + vids)

    # EndLoop outputs resolve to the type of output_value (the run object).
    for end in (a[5], b[5]):
        end.out_type_override["outputs"] = "I2ROTO_RUN"

    # serialise ----------------------------------------------------------------------------
    g.compute_order()
    links = [[lid, sid, si, did, dslot, t] for lid, sid, si, did, dslot, t in g.resolved_links()]
    state = {"lastGroupId": len(groups), "lastNodeId": ids.node, "lastLinkId": ids.link, "lastRerouteId": 0}
    wf = {
        "id": _uuid("workflow/" + ("stub" if stub else "real")),
        "revision": 0,
        "last_node_id": ids.node,
        "last_link_id": ids.link,
        "nodes": [n.to_json(None) for n in g.nodes],
        "links": links,
        "groups": groups,
        "definitions": {"subgraphs": [sg.to_json(state)]},
        "config": {},
        "extra": {"ds": {"scale": 0.55, "offset": [80, 120]}},
        "version": 0.4,
    }
    return wf


def main():
    real = build(stub=False)
    stub = build(stub=True)
    os.makedirs(os.path.join(PACK, "workflows"), exist_ok=True)
    p1 = os.path.join(PACK, "workflows", "i2_Roto_LTX25_AlphaGen_HiRes.json")
    p2 = os.path.join(PACK, "tests", "dev_stub_test.json")
    with open(p1, "w") as f:
        json.dump(real, f, indent=1)
    with open(p2, "w") as f:
        json.dump(stub, f, indent=1)
    print("wrote", p1, len(real["nodes"]), "nodes,", len(real["links"]), "links")
    print("wrote", p2, len(stub["nodes"]), "nodes,", len(stub["links"]), "links")


if __name__ == "__main__":
    main()
