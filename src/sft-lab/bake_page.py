#!/usr/bin/env python3
"""Bake the Part 3 page: splice sft-assets.json and js/engine.mjs into
src/sft-body.html between their markers. Idempotent — re-running replaces
the previously baked blocks.

Usage: python3 src/sft-lab/bake_page.py
"""
import json
import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
SRC = LAB.parent
BODY = SRC / "sft-body.html"
ASSETS = SRC / "sft-assets.json"
ENGINE = LAB / "js" / "engine.mjs"


def esm_to_iife(code: str) -> str:
    """Turn engine.mjs (export function/const) into an IIFE assigned to SFTE."""
    names = re.findall(r"^export\s+(?:function|const|var|let)\s+([A-Za-z_$][\w$]*)",
                       code, flags=re.M)
    if not names:
        sys.exit("bake: no exports found in engine.mjs")
    stripped = re.sub(r"^export\s+", "", code, flags=re.M)
    ret = ", ".join(f"{n}: {n}" for n in names)
    return f"var SFTE = (function(){{\n'use strict';\n{stripped}\nreturn {{ {ret} }};\n}})();"


def splice(html: str, start: str, end: str, payload: str) -> str:
    a = html.index(start) + len(start)
    b = html.index(end)
    return html[:a] + "\n" + payload + "\n" + html[b:]


def main() -> None:
    html = BODY.read_text(encoding="utf-8")
    assets = json.loads(ASSETS.read_text(encoding="utf-8"))
    # compact but safe inside a <script> block: escape any "</" sequence
    assets_js = "var SFT_ASSETS = " + json.dumps(
        assets, ensure_ascii=True, separators=(",", ":")
    ).replace("</", "<\\/") + ";"
    engine_js = esm_to_iife(ENGINE.read_text(encoding="utf-8"))

    html = splice(html, "/*__SFT_ASSETS_START__*/", "/*__SFT_ASSETS_END__*/", assets_js)
    html = splice(html, "/*__SFT_ENGINE_START__*/", "/*__SFT_ENGINE_END__*/", engine_js)
    BODY.write_text(html, encoding="utf-8")
    print(f"baked: assets {len(assets_js):,} B, engine {len(engine_js):,} B "
          f"-> {BODY.name} {BODY.stat().st_size:,} B")


if __name__ == "__main__":
    main()
