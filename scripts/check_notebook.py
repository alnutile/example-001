"""Validate notebook structure, embedded code, and preservation of Part B."""
import ast
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / "01_agentic_wikipedia_aimpoint_interview.ipynb", as_version=4)
original = nbformat.read(root / "reference/01_agentic_wikipedia_aimpoint_interview.ipynb", as_version=4)
nbformat.validate(notebook)
python_cells = [c for c in notebook.cells if c.cell_type == "code" and not c.source.startswith("%pip")]
for i, cell in enumerate(python_cells):
    ast.parse(cell.source, filename=f"notebook-cell-{i}")
for module in ("workflow.py", "corpus.py"):
    source = (root / "crosscheck" / module).read_text().strip()
    assert any(cell.source.strip() == source for cell in python_cells), f"Stale embedded {module}"
assert notebook.cells[-1].source.startswith(original.cells[6].source.split("> Enter your reflection here")[0])
print(f"Notebook valid: {len(python_cells)} Python cells parse; embedded modules match; Part B questions preserved.")
