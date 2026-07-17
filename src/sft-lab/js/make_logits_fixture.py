"""Export ground-truth logit vectors for the JS forward-pass parity check.

Decodes base_weights_fp16.b64.json exactly as the page will (fp16 -> fp32),
then computes RAW logits (emb @ W1 + b1 -> tanh -> h @ W2 + b2, no softmax
shift) for ~20 fixed contexts. parity.mjs compares the JS forward() output
against these; max |delta| must be < 1e-3.

Usage: python3 js/make_logits_fixture.py   (from src/sft-lab/)
Writes: js/logits_fixture.json
"""

import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, LAB)

import model as M  # noqa: E402

PROMPTS = [
    "q: what colour is the fox?\n",
    "q: where does the mole live?\n",
    "q: what does the whale eat?\n",
    "q: what colour is the wolf?\n",
    "q: what does the puma eat?\n",       # fake-entity probe context
    "q: where does the elf live?\n",
    "the fox is ",
    "the snail lives in the ",
    "the whale eats ",
    "the wolf is ",
    "the seal eats ",
    "the koi is ",
    "what colour is the swan?",           # no-newline before-beat context
    "what does the bat eat?",
    "q: what colour is the hen?\na: ",    # mid-demo context
    "q: what does the ant eat?\na: the ",
    "\n",                                  # bare start
    " ",
    "the",
    "a: the wren lives in the hedge.\n",
]


def decode_fp16(path):
    with open(path) as f:
        obj = json.load(f)
    raw = base64.b64decode(obj["b64"])
    p, off = {}, 0
    for k in obj["param_order"]:
        shape = obj["shapes"][k]
        cnt = int(np.prod(shape))
        p[k] = np.frombuffer(raw, np.float16, cnt, off).astype(np.float32).reshape(shape)
        off += cnt * 2
    return p


def raw_logits(p, ctx):
    emb = p["E"][ctx].reshape(1, -1)
    h = np.tanh(emb @ p["W1"] + p["b1"])
    return (h @ p["W2"] + p["b2"])[0]


def main():
    p = decode_fp16(os.path.join(LAB, "base_weights_fp16.b64.json"))
    cases = []
    for prompt in PROMPTS:
        ctx = M.context_ids(prompt, 26)
        cases.append({
            "prompt": prompt,
            "ctx": [int(i) for i in ctx],
            "logits": [float(x) for x in raw_logits(p, ctx)],
        })
    out_path = os.path.join(HERE, "logits_fixture.json")
    with open(out_path, "w") as f:
        json.dump({"n_cases": len(cases), "cases": cases}, f)
    print(f"wrote {out_path}: {len(cases)} contexts")


if __name__ == "__main__":
    main()
