"""Inlines assets.json into template.html -> index.html (single self-contained file)."""
from pathlib import Path

here = Path(__file__).parent
html = (here / "template.html").read_text(encoding="utf-8").replace("__ASSETS__", (here / "assets.json").read_text(encoding="utf-8"))
(here / "index.html").write_text(html, encoding="utf-8")
print("written", here / "index.html", len(html), "bytes")
