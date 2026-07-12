#!/usr/bin/env python3
"""Splice base64 font CSS into each src/*-body.html and write standalone pages
to the repo root, ready to serve directly via GitHub Pages."""
import pathlib

ROOT = pathlib.Path(__file__).parent
SRC = ROOT / "src"
FONTS = (ROOT / "fonts" / "fonts.css").read_text()

PAGES = {
    "index-body.html": "index.html",
    "ssl-body.html": "self-supervised-learning.html",
    "distillation-body.html": "knowledge-distillation.html",
}

def main():
    for src_name, out_name in PAGES.items():
        body = (SRC / src_name).read_text()
        if body.count("/*__FONTS__*/") != 1:
            raise SystemExit(f"{src_name}: expected exactly one /*__FONTS__*/ placeholder")
        built = body.replace("/*__FONTS__*/", FONTS)
        out_path = ROOT / out_name
        out_path.write_text(built)
        print(f"{src_name} -> {out_path} ({len(built):,} chars)")

if __name__ == "__main__":
    main()
