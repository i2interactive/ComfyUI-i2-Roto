"""Load the pack without ComfyUI and drive the node graph by calling node functions directly."""
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)


def load_pack():
    if "i2roto_pack" in sys.modules:
        return sys.modules["i2roto_pack"]
    spec = importlib.util.spec_from_file_location("i2roto_pack", os.path.join(PACK, "__init__.py"),
                                                  submodule_search_locations=[PACK])
    m = importlib.util.module_from_spec(spec)
    sys.modules["i2roto_pack"] = m
    spec.loader.exec_module(m)
    return m


def run_graph(cfg, verbose=True, matte_fn=None):
    """Setup -> [LoadJob -> oracle -> StoreJob] x proxy -> PlanDetail -> [..] x detail -> Assemble.

    Mirrors the workflow: the pin carried by the loop is passed along but (like the real graph) only used
    by the model, which the oracle replaces.
    """
    pack = load_pack()
    N = pack.nodes
    t0 = time.time()
    run, proxy_jobs, blank, first, info = N.I2RotoSetup().run(**cfg)
    timings = {"setup": time.time() - t0}
    pin = blank
    load, store, stub = N.I2RotoLoadJob(), N.I2RotoStoreJob(), N.I2RotoStubMatte()
    flags = {"proxy": [], "detail": []}
    t1 = time.time()
    done = None
    for i in range(proxy_jobs):
        imgs, w, h, L, first_chunk, job = load.load(run, "proxy", i)
        flags["proxy"].append(first_chunk)
        (m,) = stub.stub(imgs, job) if matte_fn is None else matte_fn(imgs, job)
        pin, done = store.store(job, m)
        del imgs, m, job  # keep peak RAM to one job (the test box has 7 GB)
    timings["proxy"] = time.time() - t1
    t2 = time.time()
    run2, detail_jobs, pinfo = N.I2RotoPlanDetail().plan([run], [done])
    run2 = run2
    timings["plan"] = time.time() - t2
    t3 = time.time()
    outs = []
    for i in range(detail_jobs):
        imgs, w, h, L, first_chunk, job = load.load(run2, "detail", i)
        flags["detail"].append(first_chunk)
        (m,) = stub.stub(imgs, job) if matte_fn is None else matte_fn(imgs, job)
        pin, d = store.store(job, m)
        outs.append(d)
        del imgs, m, job
    timings["detail"] = time.time() - t3
    t4 = time.time()
    res = N.I2RotoAssemble().assemble([run2], outs)["result"]
    timings["assemble"] = time.time() - t4
    if verbose:
        print(info)
        print(pinfo)
        print(res[3])
        print("timings", {k: round(v, 1) for k, v in timings.items()})
    return {"run": run2, "info": info, "plan_info": pinfo, "detail_jobs": detail_jobs, "proxy_jobs": proxy_jobs,
            "matte_preview": res[0], "comp_preview": res[1], "folder": res[2], "report": res[3],
            "flags": flags, "timings": timings, "first_frame": first, "blank": blank}
