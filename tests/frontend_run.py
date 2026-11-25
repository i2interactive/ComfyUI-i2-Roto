"""Load a workflow JSON in the real ComfyUI frontend (headless Chrome), check it, convert with graphToPrompt,
queue it on the server and (optionally) wait for the result.

    python tests/frontend_run.py WORKFLOW.json --server http://127.0.0.1:8199 [--setup '{"plate_path": ...}']
           [--screenshot out.png] [--wait] [--timeout 1800] [--report out.json]
"""
import argparse
import json
import sys
import time
import urllib.request
import uuid

from playwright.sync_api import sync_playwright


def http_json(url, data=None):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "null")


def patch_setup(wf, values):
    for n in wf["nodes"]:
        if n["type"] == "I2RotoSetup":
            named = n["widgets_values_named"]
            for k, v in values.items():
                if k not in named:
                    raise KeyError(k)
                named[k] = v
            n["widgets_values"] = list(named.values())
    return wf


JS_STATE = """
() => {
  const out = {nodes: [], subgraphs: []};
  const dump = (n) => ({id: n.id, type: n.type, title: n.title,
      widgets: (n.widgets || []).map(w => [w.name, (typeof w.value === 'object' && w.value !== null) ? String(w.value) : w.value]),
      inputs: (n.inputs || []).map(i => [i.name, i.type, i.link]),
      outputs: (n.outputs || []).map(o => [o.name, o.type, (o.links || []).length]),
      has_errors: !!n.has_errors});
  for (const n of app.graph.nodes) out.nodes.push(dump(n));
  const sgs = app.graph.subgraphs ? Array.from(app.graph.subgraphs.values()) : [];
  for (const sg of sgs) out.subgraphs.push({id: sg.id, name: sg.name, nodes: sg.nodes.map(dump)});
  return out;
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")
    ap.add_argument("--server", default="http://127.0.0.1:8199")
    ap.add_argument("--setup", default="{}")
    ap.add_argument("--screenshot")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--report")
    ap.add_argument("--no-queue", action="store_true")
    ap.add_argument("--set-widget", action="append", default=[],
                    help="NodeType.widget=value, applied in memory before loading (e.g. LoRA path separator)")
    a = ap.parse_args()

    wf = json.load(open(a.workflow))
    wf = patch_setup(wf, json.loads(a.setup))
    for sw in a.set_widget:
        lhs, val = sw.split("=", 1)
        typ, wname = lhs.split(".", 1)
        hit = 0
        for n in wf["nodes"]:
            if n["type"] == typ and wname in n.get("widgets_values_named", {}):
                n["widgets_values_named"][wname] = val
                n["widgets_values"] = list(n["widgets_values_named"].values())
                hit += 1
        if not hit:
            raise SystemExit(f"--set-widget: no {typ}.{wname}")
    expected = {}
    for n in wf["nodes"]:
        expected[("top", n["id"])] = (n["type"], n.get("widgets_values_named", {}))
    for sg in wf.get("definitions", {}).get("subgraphs", []):
        for n in sg["nodes"]:
            expected[(sg["id"], n["id"])] = (n["type"], n.get("widgets_values_named", {}))

    result = {"workflow": a.workflow, "console": [], "page_errors": []}
    # Avoid first-run dialogs.
    for k, v in {"Comfy.TutorialCompleted": True, "Comfy.Workflow.ShowMissingNodesWarning": True,
                 "Comfy.Workflow.ShowMissingModelsWarning": True}.items():
        try:
            http_json(f"{a.server}/settings/{k}", v)
        except Exception:
            pass
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 2400, "height": 1300})
        page.on("console", lambda m: result["console"].append([m.type, m.text]))
        page.on("pageerror", lambda e: result["page_errors"].append(str(e)))
        page.goto(a.server, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_function("() => window.app && window.app.graph && window.app.vueAppReady !== false",
                               timeout=180000)
        page.wait_for_timeout(3000)
        result["frontend_version"] = page.evaluate("() => window.__COMFYUI_FRONTEND_VERSION__ || null")
        n_console_before = len(result["console"])
        page.evaluate("async (wf) => { await app.loadGraphData(wf); }", wf)
        page.wait_for_timeout(4000)
        # Dialogs (missing nodes / models) show as PrimeVue dialogs.
        result["dialogs"] = page.evaluate(
            "() => Array.from(document.querySelectorAll('.p-dialog')).map(d => d.innerText.slice(0, 2000))")
        result["console_after_load"] = result["console"][n_console_before:]
        state = page.evaluate(JS_STATE)
        result["graph_state"] = state
        # Compare widget values the frontend ended up with to the JSON's named values.
        mism = []
        live = {}
        for n in state["nodes"]:
            live[("top", str(n["id"]))] = n
        for sg in state["subgraphs"]:
            for n in sg["nodes"]:
                live[(sg["id"], str(n["id"]))] = n
        for key, (typ, named) in expected.items():
            n = live.get((key[0], str(key[1])))
            if n is None:
                mism.append(f"{key} {typ}: node missing in frontend")
                continue
            if n["type"] != typ:
                mism.append(f"{key}: type {n['type']} != {typ}")
            wv = {w[0]: w[1] for w in n["widgets"]}
            for k, v in named.items():
                if k not in wv:
                    mism.append(f"{key} {typ}: widget {k} not on node (has {list(wv)})")
                elif wv[k] != v and not (isinstance(v, (int, float)) and isinstance(wv[k], (int, float))
                                         and abs(wv[k] - v) < 1e-9):
                    mism.append(f"{key} {typ}: widget {k} = {wv[k]!r}, expected {v!r}")
            extra = [k for k in wv if k not in named and typ not in ("PreviewAny",)]
            if extra:
                result.setdefault("extra_widgets", []).append(f"{key} {typ}: {extra}")
        result["widget_mismatches"] = mism
        if a.screenshot:
            page.evaluate("""() => { const c = app.canvas; const ns = app.graph.nodes; if (!c || !c.ds || !ns.length) return;
                let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
                for (const n of ns) { x0=Math.min(x0,n.pos[0]); y0=Math.min(y0,n.pos[1]-80); x1=Math.max(x1,n.pos[0]+n.size[0]); y1=Math.max(y1,n.pos[1]+n.size[1]); }
                const W = c.canvas.width / (window.devicePixelRatio||1), H = c.canvas.height / (window.devicePixelRatio||1);
                const sc = Math.min((W-140)/(x1-x0), (H-120)/(y1-y0));
                c.ds.scale = sc; c.ds.offset = [100/sc - x0, 80/sc - y0]; c.setDirty(true, true); }""")
            page.wait_for_timeout(1500)
            page.screenshot(path=a.screenshot)
        api = page.evaluate("async () => { const r = await app.graphToPrompt(); return {output: r.output, workflow: r.workflow}; }")
        result["api_prompt"] = api["output"]
        browser.close()

    if a.no_queue:
        print("frontend version:", result.get("frontend_version"), "mismatches:", mism, "dialogs:", result["dialogs"])
        return 0
    client_id = str(uuid.uuid4())
    status, resp = http_json(f"{a.server}/prompt", {"prompt": api["output"], "client_id": client_id,
                                                     "extra_data": {"extra_pnginfo": {"workflow": api["workflow"]}}})
    result["queue_status"] = status
    result["queue_response"] = resp
    print("frontend version:", result.get("frontend_version"))
    print("dialogs:", result["dialogs"])
    print("page errors:", result["page_errors"])
    errs = [c for c in result["console_after_load"] if c[0] in ("error", "warning")]
    print("console errors/warnings after load:", json.dumps(errs, indent=1)[:4000])
    print("widget mismatches:", json.dumps(mism, indent=1))
    print("extra widgets:", json.dumps(result.get("extra_widgets", []), indent=1))
    print("queue:", status, json.dumps(resp)[:3000])
    if status == 200 and a.wait:
        pid = resp["prompt_id"]
        t0 = time.time()
        while time.time() - t0 < a.timeout:
            _, h = http_json(f"{a.server}/history/{pid}")
            if h and pid in h and h[pid].get("status", {}).get("completed") is not None and \
                    h[pid]["status"].get("status_str") in ("success", "error"):
                result["history"] = h[pid]
                break
            time.sleep(3)
        hist = result.get("history")
        if hist is None:
            print("TIMEOUT waiting for execution")
        else:
            st = hist["status"]
            print("execution:", st.get("status_str"), "in", round(time.time() - t0, 1), "s")
            for m in st.get("messages", []):
                if m[0] in ("execution_error", "execution_interrupted"):
                    print(json.dumps(m, indent=1)[:6000])
            result["exec_seconds"] = time.time() - t0
    if a.report:
        with open(a.report, "w") as f:
            json.dump(result, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
