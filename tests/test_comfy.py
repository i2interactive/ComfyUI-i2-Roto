"""Tests 2 and 3: real ComfyUI + real frontend (headless Chrome).

Needs a running server: python3 main.py --cpu --port 8199 with I2ROTO_DEV=1 and this pack in custom_nodes,
plus (for test 3) zero-byte placeholder model files.

    python tests/test_comfy.py stub      # test 2: dev_stub_test.json, native multi-tile multi-chunk + proxy only
    python tests/test_comfy.py real      # test 3: shipped workflow, must pass validation and fail at model load
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
WORK = os.environ.get("I2ROTO_WORK", "/home/claude/i2roto/_work")
COMFY = os.environ.get("COMFY_DIR", "/home/claude/research/comfy")
SERVER = os.environ.get("COMFY_SERVER", "http://127.0.0.1:8199")

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from i2roto.synth import Synth  # noqa: E402

PLATE_DIR = os.path.join(WORK, "plate3k")
FAIL = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def frontend(workflow, setup, report, screenshot=None, wait=True, extra=()):
    cmd = [sys.executable, os.path.join(HERE, "frontend_run.py"), workflow, "--server", SERVER,
           "--setup", json.dumps(setup), "--report", report, "--timeout", "2400"]
    if wait:
        cmd.append("--wait")
    if screenshot:
        cmd += ["--screenshot", screenshot]
    cmd += list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)
    print(p.stdout[-3000:])
    if p.returncode != 0:
        print(p.stderr[-3000:])
    return json.load(open(report))


def common_checks(tag, r):
    check(f"{tag}: frontend loaded without dialogs / page errors", not r["dialogs"] and not r["page_errors"],
          f"dialogs {r['dialogs']}, page errors {r['page_errors']}")
    errs = [c for c in r["console_after_load"] if c[0] == "error"]
    check(f"{tag}: no console errors after load", not errs, json.dumps(errs)[:500])
    check(f"{tag}: every widget value as authored", not r["widget_mismatches"], json.dumps(r["widget_mismatches"])[:800])
    check(f"{tag}: server accepted the prompt at validation", r["queue_status"] == 200 and
          not r["queue_response"].get("node_errors"), json.dumps(r["queue_response"])[:800])


def verify_outputs(folder, n, first_number=1001):
    s = Synth.load(os.path.join(PLATE_DIR, "i2roto_synth.json"))
    files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    maes = []
    for i, f in enumerate(files):
        a = cv2.imread(os.path.join(folder, f), cv2.IMREAD_UNCHANGED)
        assert a.dtype == np.uint16 and a.ndim == 2
        maes.append(float(np.abs(a.astype(np.float32) / 65535 - s.alpha(i)).mean()))
    ok_names = files == [f"shot_matte.{first_number + i}.png" for i in range(n)]
    return files, maes, ok_names


def stub():
    wf = os.path.join(HERE, "dev_stub_test.json")
    plate = os.path.join(PLATE_DIR, "shot.####.png")
    res = {}
    for tag, extra_setup in [("native", {"detail": "native (max quality)", "window": "1280x704 (low VRAM, faster)"}),
                             ("proxy", {"detail": "proxy only (fastest)"})]:
        out_rel = f"i2roto_test_stub_{tag}"
        out = os.path.join(COMFY, "output", out_rel)
        if os.path.isdir(out):
            for f in os.listdir(out):
                os.remove(os.path.join(out, f))
        setup = {"plate_path": plate, "chunk_frames": 17, "output_folder": out_rel}
        setup.update(extra_setup)
        r = frontend(wf, setup, os.path.join(WORK, f"comfy_stub_{tag}.json"),
                     screenshot=os.path.join(HERE, "graph_stub.png") if tag == "native" else None)
        common_checks(f"stub {tag}", r)
        h = r.get("history", {})
        st = h.get("status", {})
        check(f"stub {tag}: executed end to end", st.get("status_str") == "success",
              f"{st.get('status_str')} in {r.get('exec_seconds', 0):.0f}s")
        texts = {k: v.get("text") for k, v in h.get("outputs", {}).items() if "text" in v}
        report = next((t[0] for t in texts.values() if t and t[0].startswith("Wrote")), "")
        plan = next((t[0] for t in texts.values() if t and t[0].startswith("Detail pass")), "")
        print(plan)
        print(report)
        files, maes, ok_names = verify_outputs(out, 41)
        check(f"stub {tag}: 41 frames shot_matte.1001..1041.png on disk", len(files) == 41 and ok_names,
              f"{len(files)} files")
        res[tag] = {"plan": plan, "report": report, "mae_mean": float(np.mean(maes)), "mae_max": float(np.max(maes)),
                    "exec_s": r.get("exec_seconds")}
        if tag == "native":
            check("stub native: multi-tile multi-chunk detail pass ran",
                  "Detail loop returned 1 item" in report and " 0 jobs" not in report, plan)
            check("stub native: matte MAE vs native ground truth < 0.004", max(maes) < 0.004,
                  f"mean {np.mean(maes):.6f} max {max(maes):.6f}")
        else:
            check("stub proxy: detail loop ran zero iterations and Assemble still executed",
                  "Detail loop returned 0 item(s)" in report and "0 jobs" in report, report.splitlines()[-1] if report else "")
            check("stub proxy: proxy-only matte is plausible (MAE < 0.02)", max(maes) < 0.02,
                  f"mean {np.mean(maes):.6f} max {max(maes):.6f}")
    with open(os.path.join(WORK, "results", "comfy_stub.json"), "w") as f:
        json.dump(res, f, indent=1)


def real():
    wf = os.path.join(PACK, "workflows", "i2_Roto_LTX25_AlphaGen_HiRes.json")
    shipped = json.load(open(wf))
    lora = [n for n in shipped["nodes"] if n["type"] == "LoraLoaderModelOnly"][0]
    check("real: shipped LoRA value is the Windows subfolder path",
          lora["widgets_values_named"]["lora_name"] == "LTX\\ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors",
          lora["widgets_values_named"]["lora_name"])
    plate = os.path.join(PLATE_DIR, "shot.####.png")
    r = frontend(wf, {"plate_path": plate}, os.path.join(WORK, "comfy_real.json"),
                 screenshot=os.path.join(HERE, "graph_real.png"),
                 extra=["--set-widget", "LoraLoaderModelOnly.lora_name=LTX/ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"])
    common_checks("real", r)
    api = r["api_prompt"]
    types = sorted({v["class_type"] for v in api.values()})
    check("real: API prompt has 2 flattened copies of the matte subgraph (10 nodes each)",
          sum(1 for k in api if ":" in k) == 20, f"{len(api)} nodes; types {types}")
    h = r.get("history", {})
    st = h.get("status", {})
    errs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
    e = errs[0][1] if errs else {}
    print("EXECUTION ERROR:", json.dumps({k: e.get(k) for k in ("node_id", "node_type", "exception_type",
                                                                 "exception_message")}, indent=1))
    loaders = {"UNETLoader", "VAELoader", "CLIPLoader", "LoraLoaderModelOnly"}
    check("real: failed at model load (zero-byte placeholder), not earlier",
          st.get("status_str") == "error" and e.get("node_type") in loaders,
          f"{e.get('node_type')} #{e.get('node_id')}: {e.get('exception_type')}: {str(e.get('exception_message')).strip()}")
    with open(os.path.join(WORK, "results", "comfy_real.json"), "w") as f:
        json.dump({"error": {k: e.get(k) for k in ("node_id", "node_type", "exception_type", "exception_message")},
                   "frontend_version": r.get("frontend_version"), "nodes_in_api_prompt": len(api)}, f, indent=1)


if __name__ == "__main__":
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    for what in sys.argv[1:] or ["stub", "real"]:
        {"stub": stub, "real": real}[what]()
    print(f"\n{len(FAIL)} failed: {FAIL}")
    sys.exit(1 if FAIL else 0)
