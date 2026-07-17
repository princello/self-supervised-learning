"""Export base weights as fp16 base64 (the form that ships inside the page).

Writes base_weights_fp16.b64.json and prints a size report. Also verifies the
fp16 roundtrip: decodes the b64 back to weights and re-runs the full fact-
completion eval so we know quantization does not change page behaviour.

Usage: python3 export_b64.py [--weights base_weights.json]
"""

import argparse
import base64
import hashlib
import json
import os

import numpy as np

import model as M
from eval_base import eval_facts

HERE = os.path.dirname(os.path.abspath(__file__))


def export(weights="base_weights.json", out="base_weights_fp16.b64.json"):
    p, cfg, meta = M.load_json(os.path.join(HERE, weights))
    blob = b"".join(
        np.ascontiguousarray(p[k].astype(np.float16)).tobytes() for k in M.PARAM_ORDER
    )
    b64 = base64.b64encode(blob).decode("ascii")
    obj = {
        "config": cfg,
        "param_order": M.PARAM_ORDER,
        "shapes": {k: list(p[k].shape) for k in M.PARAM_ORDER},
        "dtype": "float16",
        "encoding": "base64 of concatenated little-endian fp16 arrays in param_order",
        "sha256_fp32_source": meta.get("weights_sha256_fp32", ""),
        "sha256_b64": hashlib.sha256(b64.encode()).hexdigest(),
        "b64": b64,
    }
    out_path = os.path.join(HERE, out)
    with open(out_path, "w") as f:
        json.dump(obj, f)

    n = M.n_params(p)
    print("size report:")
    print(f"  params:          {n}")
    print(f"  fp16 raw bytes:  {len(blob)}")
    print(f"  base64 chars:    {len(b64)}")
    print(f"  json file bytes: {os.path.getsize(out_path)}")

    # roundtrip: decode b64 -> fp16 -> fp32 params, verify facts still 100%
    raw = base64.b64decode(obj["b64"])
    q, off = {}, 0
    for k in M.PARAM_ORDER:
        shape = obj["shapes"][k]
        cnt = int(np.prod(shape))
        q[k] = np.frombuffer(raw, np.float16, cnt, off).astype(np.float32).reshape(shape)
        off += cnt * 2
    max_err = max(float(np.abs(q[k] - p[k]).max()) for k in M.PARAM_ORDER)
    print(f"  max |fp16 - fp32| weight error: {max_err:.6f}")
    res = eval_facts(q, cfg, verbose=False)
    print(f"  fact completion with fp16-decoded weights: "
          f"{res['n_ok']}/{res['n_total']} = {res['accuracy']:.1%}")
    return obj, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="base_weights.json")
    ap.add_argument("--out", default="base_weights_fp16.b64.json")
    args = ap.parse_args()
    export(args.weights, args.out)
