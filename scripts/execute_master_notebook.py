"""
Execute Nexora_FinalNotebook.ipynb end-to-end and bake in all outputs, tables, and figures.
"""

import sys
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
NB_PATH = BASE_DIR / "notebooks" / "Nexora_FinalNotebook.ipynb"

print(f"[RUN] Loading notebook: {NB_PATH}")
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(
    nb,
    timeout=300,
    kernel_name="python3",
    resources={"metadata": {"path": str(BASE_DIR / "notebooks")}}
)

print("[RUN] Executing all cells in Nexora_FinalNotebook.ipynb...")
client.execute()
print("[SUCCESS] All cells executed successfully.")

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"[OK] Saved fully baked-in executed notebook -> {NB_PATH.name} ({NB_PATH.stat().st_size:,} bytes)")
