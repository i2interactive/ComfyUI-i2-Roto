# Test results

Run on 2026-11-13 in a Linux sandbox: 2 CPU cores, 7 GB RAM, no GPU, Python 3.13, ComfyUI 0.45.0 (HEAD checkout),
**frontend package 1.59.1** (newer than the target's 1.58.5), Chrome 153 headless via Playwright.

**No real model ran here.** Everything model-shaped was replaced by an *oracle*: the analytic ground-truth alpha of
the synthetic plate for exactly the job's region and frames, pushed through the same geometry as Load Job
(area resize for the proxy, crop for tiles, edge padding to multiples of 32). These tests check the plumbing,
geometry, loops, planner and assembler. They say nothing about the quality or speed of the real model.
The timings below are this CPU sandbox, for the pack's own code only.

How to re-run (data goes to `$I2ROTO_WORK`, default `/home/claude/i2roto/_work`):

```
python tests/test_pipeline.py chunks options video seams guided sixk     # test 1 (about 30 min here)
python tests/test_pipeline.py exr                                       # see EXR notes
python tests/test_comfy.py stub real                                    # tests 2 + 3, needs the server
```

## Test 1: pipeline without ComfyUI (`tests/test_pipeline.py`)

Synthetic plate (`i2roto/synth.py`): textured moving background; moving figure with a hard-edged body and head,
a soft (motion-blur-like, ~50 px falloff) arm, 36 hair strands 1 to 3 px wide at native, and a semi-transparent
(alpha 0.5) veil. Alpha is analytic at any resolution. Generated frame by frame.

### Chunking (1920x1088 plate, window 1920x1088 so the proxy is native and every frame can be checked exactly)

chunk_frames 49. For each case: every frame stored exactly once (cache write log), output files exactly the expected
names, each output frame matches its own ground-truth frame (max frame MAE 1.97e-07) and is further from both
neighbours' ground truth (identity check), no warnings. `start_number` was set to 5 to prove it is ignored for
numbered sequences.

| n | start_frame | chunks | chunk lengths | stored / unique | files | is_first flags |
|---|---|---|---|---|---|---|
| 20 | 0 | 1 | 25 | 20 / 20 | 1001..1020 | T |
| 49 | 0 | 1 | 49 | 49 / 49 | 1001..1049 | T |
| 50 | 0 | 2 | 49, 49 | 50 / 50 | 1001..1050 | T F |
| 97 | 0 | 2 | 49, 49 | 97 / 97 | 1001..1097 | T F |
| 130 | 0 | 3 | 49, 49, 49 | 130 / 130 | 1001..1130 | T F F |
| 50 | 7 | 2 | 49, 49 | 50 / 50 | 1008..1057 | T F |

All PASS.

### 6144x3240, 20 frames, 16-bit PNG sequence (defaults: window 1920x1088, chunk_frames 49 -> one 25-frame chunk)

| detail | tiles / jobs | working res | MAE vs native GT | edge-band MAE | Assemble s/frame (this CPU) |
|---|---|---|---|---|---|
| native | 12 / 12 | 6144x3240 | 0.000020 | 0.0000050 | 2.43 |
| 2x proxy | 6 / 6 | 3840x2025 | 0.000182 | 0.002837 | 3.48 (guided filter) |
| proxy only | 0 / 0 | 1920x1012 | 0.000483 | 0.007399 | 2.55 (guided filter) |

- native MAE < 0.004: PASS (0.000020; vs the clip-remapped ground truth it is 3e-7, the rest is the
  0.005/0.995 black/white clip).
- native edge-band MAE clearly better than proxy only (< 0.5x): PASS (0.0000050 vs 0.0074).
- Output: 20 files `shot_matte.1001.png .. shot_matte.1020.png`, uint16, single channel, 6144x3240: PASS for all three modes.
- "Edge band" here = pixels within 24 px of any ground-truth pixel with 0.002 < alpha < 0.998.
- Wall time of the whole native run here: setup 2.5 s, proxy loop 33 s, plan 0.7 s, detail loop 333 s (12 jobs x 25
  frames, mostly re-reading 6K PNGs and rendering the oracle), assemble 49 s.

Seams. The spec's literal criterion ("max abs error along overlap centre lines not above the band average") is not
a meaningful statistic: a maximum is almost always above an average. With exact oracle tiles the numbers are:
centre-line MAE 4.77e-06 (69,763 band pixels on centre lines), max 8.7e-04; band MAE 4.99e-06, band max 2.2e-03.
So the **literal** test would FAIL (8.7e-04 > 5.0e-06). What I assert instead: centre-line MAE <= band MAE and
centre-line max <= band max (PASS). Because oracle tiles agree perfectly, that alone proves little, so there is a
second, stronger test:

### Seam stress (3072x1620, 9 frames)

Proxy oracle = flat 0.5 (band covers the whole frame, tiles cover everything); each detail tile returns a flat
0.45 or 0.55 depending on tile parity, so neighbouring tiles disagree by 0.10. The written matte is then exactly the
tile blend. Max adjacent-pixel step anywhere in the frame: **0.00034** (a hard seam would be 0.10), 12 tiles.
Tile interiors keep their own value (max deviation 3.8e-06). PASS.

### Guided upscale (proxy only, 3072x1620, 9 frames)

Edge-band MAE with the guided filter 0.00893 vs plain bicubic 0.01338; whole-frame MAE 0.000786 vs 0.001169. PASS
(guided is better here; on a synthetic plate whose matte edges coincide with colour edges, which favours it).

### Video input vs sequence input (6144x3240, 20 frames, BT.709-tagged)

Read via Setup + Load Job with start_frame 11 / frame_count 6 (so the decoder must seek into the middle of the file),
plus random access to frames [0, 5, 7, 8, 13, 19, 19] and a lone seek to frame 13.

| file | proxy-job MAE vs PNG | own-frame MAE max vs neighbour-frame MAE min | native random access MAE | result |
|---|---|---|---|---|
| ProRes 422 HQ 10-bit .mov (intra) | 0.00026 | 0.00026 < 0.0332 | 0.0006 per frame | PASS |
| H.264 8-bit 4:2:0 .mp4, GOP 8, 2 B-frames | 0.00495 | 0.00496 < 0.0338 | ~0.0052 per frame | PASS |

Probe: 6144x3240, 20 frames, 24 fps for both.

### Options smoke (1536x864, 12 frames, `2x proxy`, chunk 9)

TIFF 16-bit output, invert, edge shift +2.5 px, garbage matte (single PNG, left half white, path given in quotes),
write_rgba, plate_color `log -> auto normalize`. Checks: 12 uint16 TIFFs at 1536x864; garbage matte zeroes the right
half and invert is applied; the spread grows the matte; 12 RGBA 16-bit PNGs (864x1536x4). All PASS. (On this small
plate 2x proxy already reaches native, d = 1.0, so the d < 1 path is covered by the 6K 2x run above.)

### EXR

- The sandbox's OpenCV (5.0.0) has **no OpenEXR**. I checked the official `opencv-python-headless` 5.0.0.93 pip wheel
  too: also built without OpenEXR. The 4.14.0 wheel has it.
- With the system OpenCV 5: an EXR plate gives the clear error (PASS); EXR output works through the pack's own
  minimal EXR writer (PASS), and those files read back correctly in OpenCV 4.14 with OpenEXR: float32 single channel,
  MAE 6.3e-07 vs ground truth (PASS).
- With OpenCV 4.14 (installed with `pip --target` into `_work/cv2exr4`, used only via `PYTHONPATH` for this test,
  not installed system-wide): linear-light half-float EXR plate in (`linear (EXR) -> sRGB`), EXR half out.
  6 files `shot_matte.0001..0006.exr` (sequence numbers kept), float32 single channel, matte MAE 1e-06 (PASS);
  linear -> sRGB conversion matches the display plate, MAE 3.3e-05 (PASS); built-in writer round-trip max error
  2.4e-04 (half precision) (PASS).

## Test 2: real ComfyUI, CPU, stub model (`tests/test_comfy.py stub`)

Server: `python3 main.py --cpu --port 8199`, `I2ROTO_DEV=1`, pack symlinked into `custom_nodes`.
`tests/dev_stub_test.json` loaded in the real frontend (headless Chrome), `app.loadGraphData` ->
`app.graphToPrompt` -> POST `/prompt` -> poll `/history`. The stub graph keeps the same loop and subgraph structure
as the real one: the matte step is a subgraph ("Matte one job (STUB, dev test only)") instantiated in both loops,
so subgraph flattening inside StartLoop/EndLoop bodies was executed for real.

Plate: 3072x1620, 41 frames, 16-bit PNG, chunk_frames 17 (3 chunks).

| case | frontend | validation | execution | output |
|---|---|---|---|---|
| native, window 1280x704 | no dialogs, no console errors, all widget values as authored | accepted, no node_errors | success, 259 s | 36 detail jobs (12 tiles x 3 chunks); 41 files 1001..1041; MAE vs native GT 0.000024 |
| proxy only, window 1920x1088 | same | accepted | success, 63 s | detail loop ran **zero iterations**; Assemble still ran and received an empty list ("Detail loop returned 0 item(s)"); 41 files; MAE 0.0005 |

So risky area 1 is verified on this ComfyUI: a node downstream of an EndLoop whose loop ran zero times still
executes when it is `INPUT_IS_LIST` (EndLoop's output is `[]`). Screenshot: `tests/graph_stub.png`.

## Test 3: real workflow validation (`tests/test_comfy.py real`)

Zero-byte placeholders created in the test ComfyUI:
`models/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`,
`models/loras/LTX/ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors`, `models/vae/ltx-2.5-video-vae-bf16.safetensors`,
`models/vae/ltx-2.5-audio-vae-bf16.safetensors`, `models/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`.

The shipped workflow has the LoRA as `LTX\ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors` (Windows path, as on the
target machine; checked by the test). On Linux ComfyUI lists it as `LTX/...`, so **for this run only** the value was
patched in memory to `LTX/ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors` before loading (`--set-widget`); the
shipped file is unchanged. Setup's `plate_path` was set to the 3K test plate.

- Frontend 1.59.1: no missing-node / missing-model dialogs, no page errors, no console errors after load, every
  widget value (top level and inside the subgraph) as authored.
- `graphToPrompt`: 54 API nodes, including two flattened copies (`30:1..30:10`, `39:1..39:10`) of the
  "Matte one job (LTX Alpha Gen)" subgraph.
- POST `/prompt`: HTTP 200, `node_errors: {}` (accepted at validation, including loop pairing).
- Execution then failed at model load, as expected. Exact error recorded:

```
node 18 (VAELoader): safetensors._safetensors_rust.SafetensorError: Error while deserializing header: header too small
```

Screenshot: `tests/graph_real.png`.
