"""Point every Colab badge / clone URL at your GitHub repo.

Usage:  python scripts/set_repo.py your-user/llm-knowledge-workshop
"""
import sys
from pathlib import Path

PLACEHOLDER = "YOUR-GITHUB-USER/llm-knowledge-workshop"
ROOT = Path(__file__).resolve().parents[1]

if len(sys.argv) != 2 or "/" not in sys.argv[1]:
    sys.exit(__doc__)
slug = sys.argv[1].strip("/")
targets = [ROOT / "mkdocs.yml", *ROOT.glob("docs/**/*.md"), *ROOT.glob("notebooks/*.ipynb"), *ROOT.glob("scripts/make_*.py")]
for f in targets:
    text = f.read_text()
    if PLACEHOLDER in text:
        f.write_text(text.replace(PLACEHOLDER, slug))
        print("updated", f.relative_to(ROOT))
