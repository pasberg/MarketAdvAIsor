"""Build the static GitHub Pages site from the mockup page.

The mockup (mockup/index.html) is written as page content without <html>/<head>
tags (the artifact viewer adds them); this wraps it in a full document.
"""
from __future__ import annotations

import argparse
from pathlib import Path

HEAD = """<!doctype html>
<html lang="sv">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Förslag på aktier, index och råvaror per tidshorisont — teknisk analys, Fibonacci och momentum.">
<style>html{color-scheme:light dark}body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>
</head>
<body>
"""
TAIL = "\n</body>\n</html>\n"


def build(src: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HEAD + src.read_text(encoding="utf-8") + TAIL, encoding="utf-8")
    (out.parent / ".nojekyll").write_text("", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="mockup/index.html")
    ap.add_argument("--out", default="site/index.html")
    a = ap.parse_args()
    build(Path(a.src), Path(a.out))
    print(f"{a.out} byggd från {a.src}")


if __name__ == "__main__":
    main()
