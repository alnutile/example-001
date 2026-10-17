"""Render the reference architecture as a one-page vector PDF."""

from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/crosscheck-architecture.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
c = canvas.Canvas(str(OUT), pagesize=(1200, 800))
c.setTitle("CrossCheck - reference architecture")
c.setAuthor("Aimpoint assessment working draft")
INK = HexColor("#142D40")
MUTED = HexColor("#526779")
TEAL = HexColor("#087E8B")
BLUE = HexColor("#3958A5")
BORDER = HexColor("#CDDAE2")


def text(x, y, value, size=11, color=INK, bold=False):
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.drawString(x, y, value)


def box(x, y, w, h, title, lines, color=TEAL, fill="#FFFFFF"):
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(BORDER)
    c.setLineWidth(1)
    c.roundRect(x, y, w, h, 9, fill=1, stroke=1)
    c.setFillColor(color)
    c.roundRect(x, y+h-6, w, 6, 3, fill=1, stroke=0)
    text(x+13, y+h-28, title, 13, bold=True)
    for i, line in enumerate(lines):
        text(x+13, y+h-47-i*15, line, 10.5, MUTED)


def arrow(points, color=MUTED, dashed=False):
    c.setStrokeColor(color)
    c.setLineWidth(1.4)
    c.setDash(5, 4) if dashed else c.setDash()
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path)
    c.setDash()
    x, y = points[-1]
    px, py = points[-2]
    dx, dy = x-px, y-py
    length = (dx*dx+dy*dy)**0.5
    dx, dy = dx/length, dy/length
    tip = c.beginPath()
    tip.moveTo(x, y)
    tip.lineTo(x-7*dx+3.5*dy, y-7*dy-3.5*dx)
    tip.lineTo(x-7*dx-3.5*dy, y-7*dy+3.5*dx)
    tip.close()
    c.setFillColor(color)
    c.drawPath(tip, stroke=0, fill=1)


c.setFillColor(HexColor("#F7FAFC"))
c.rect(0, 0, 1200, 800, fill=1, stroke=0)
text(40, 746, "CrossCheck", 34, bold=True)
text(40, 717, "Cited claim verification over Apollo and Space Race Wikipedia articles", 16, MUTED)
text(840, 754, "REFERENCE ARCHITECTURE", 11, TEAL, True)
text(840, 735, "Implementation draft | Databricks Free Edition", 11, MUTED)
text(840, 717, "Live model evaluation pending", 11, MUTED)

text(40, 665, "01  PREPARE A SMALL, REUSABLE CORPUS", 11, TEAL, True)
box(40, 550, 250, 94, "WikipediaLoader", ["8 topic queries, one string per call", "5 results/query; 40 article global cap", "Deduplicate by article URL"])
box(330, 550, 250, 94, "JSON corpus snapshot", ["Article text + title + source URL", "Content hash and retrieval timestamp", "Reuse snapshot for demo and evaluation"])
box(620, 550, 250, 94, "Passages + embeddings", ["1,000 characters; 150 overlap", "DatabricksEmbeddings", "Configured embedding endpoint"])
box(910, 550, 250, 94, "FAISS dense index", ["In-memory index of short passages", "Reuse within the notebook session", "Keep source metadata attached"])
for left in (290, 580, 870):
    arrow([(left, 597), (left+40, 597)])

text(40, 500, "02  LANGGRAPH WORKFLOW", 11, BLUE, True)
arrow([(1035, 550), (1035, 520), (505, 520), (505, 469)], TEAL, True)
text(655, 527, "retrieval reads the same index", 10, TEAL)
box(40, 375, 170, 94, "Assertions", ["User-written paragraph", "Claims to check", "Example: mission dates"], BLUE)
box(230, 375, 170, 94, "Claim extractor", ["ChatDatabricks", "Atomic claims + queries", "Author's custom prompt"], BLUE)
box(420, 375, 170, 94, "Retriever", ["Claim + entity query", "Rank fusion; 4 passages", "Max 2 per source"], BLUE)
box(610, 375, 170, 94, "Verifier + checks", ["ChatDatabricks + Python", "Verdict, exact quotes, IDs", "Validate provenance"], BLUE)
box(800, 375, 170, 94, "Quality metrics", ["Citation coverage", "Abstentions and errors", "Potential conflicts"], BLUE)
box(990, 375, 170, 94, "Cited report", ["Deterministic HTML", "Claim / verdict / quote", "Links from source data"], BLUE)
for left in (210, 400, 590, 780, 970):
    arrow([(left, 422), (left+20, 422)], BLUE)

box(515, 220, 260, 96, "One bounded retry", ["Only insufficient evidence + a new query", "Search FAISS; append up to 2 passages", "Reverify, then finish regardless of verdict"], BLUE, "#EFF4FF")
arrow([(695, 375), (695, 316)], BLUE)
text(705, 348, "if useful", 10, BLUE)
arrow([(775, 267), (885, 267), (885, 375)], BLUE)
text(812, 279, "finish", 10, BLUE)
text(40, 333, "State", 11, BLUE, True)
text(40, 314, "claims | evidence | verdicts | query history", 11, MUTED)
text(40, 296, "retry count | metrics | trace | report", 11, MUTED)
text(40, 263, "Input questions need explicit assertions first.", 10.5, MUTED)
text(40, 245, "At most 6 claims; at most one follow-up per claim.", 10.5, MUTED)

text(40, 185, "03  DIAGNOSTIC EVALUATION AND INTERPRETATION", 11, TEAL, True)
box(40, 60, 320, 106, "Hand-labelled cases", ["Six candidate claims; labels supplied by author", "Enter the same graph at retrieval (skip extractor)", "Score accuracy and per-class results", "Inspect extraction separately on the demo"], TEAL)
arrow([(360, 113), (392, 113), (392, 350), (505, 350), (505, 375)], TEAL, True)
box(420, 60, 355, 106, "What the checks establish", ["Exact quotes must occur in a retrieved passage.", "Supported/refuted results require cited evidence.", "Invalid output earns no evaluation credit.", "Quote presence does not establish entailment."], TEAL)
box(815, 60, 345, 106, "How to read the results", ["Coverage is enforced; also inspect abstention rate.", "No decided claims means coverage is N/A.", "Conflicting stances flag potential disagreement.", "Shared Wikipedia sources are not independence."], TEAL)
text(40, 25, "Two vector queries = multi-query dense retrieval. Evaluation uses the frozen corpus; it is not run recursively inside the application.", 10, MUTED)
c.showPage()
c.save()
print(OUT)
