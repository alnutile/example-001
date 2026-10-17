"""Render the reference architecture as a one-page vector PDF."""

from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/crosscheck-architecture.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
H = 920  # page height; rows above the evaluation band are shifted up by D
D = 120
c = canvas.Canvas(str(OUT), pagesize=(1200, H))
c.setTitle("CrossCheck - reference architecture")
c.setAuthor("Alfred Nutile")
INK = HexColor("#142D40")
MUTED = HexColor("#526779")
TEAL = HexColor("#087E8B")
BLUE = HexColor("#3958A5")
BORDER = HexColor("#CDDAE2")
GOLD = HexColor("#B7791F")


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
c.rect(0, 0, 1200, H, fill=1, stroke=0)
text(40, 746+D, "CrossCheck", 34, bold=True)
text(40, 717+D, "Cited claim verification over Apollo and Space Race Wikipedia articles", 16, MUTED)
text(800, 754+D, "REFERENCE ARCHITECTURE", 11, TEAL, True)
text(800, 735+D, "Databricks Free Edition | serverless notebook", 11, MUTED)
text(800, 717+D, "Live golden set: 6/6 correct | 100% citation coverage", 11, MUTED)

text(40, 665+D, "01  PREPARE A SMALL, REUSABLE CORPUS", 11, TEAL, True)
box(40, 550+D, 250, 94, "WikipediaLoader", ["8 topic queries, one string per call", "5 results/query; 40 article global cap", "Deduplicate by article URL"])
box(330, 550+D, 250, 94, "JSON corpus snapshot", ["Article text + title + source URL", "Content hash and retrieval timestamp", "Reuse snapshot for demo and evaluation"])
box(620, 550+D, 250, 94, "Passages + embeddings", ["1,000 characters; 150 overlap", "DatabricksEmbeddings", "databricks-bge-large-en"])
box(910, 550+D, 250, 94, "FAISS dense index", ["In-memory index of short passages", "Reuse within the notebook session", "Keep source metadata attached"])
for left in (290, 580, 870):
    arrow([(left, 597+D), (left+40, 597+D)])

text(40, 500+D, "02  LANGGRAPH WORKFLOW", 11, BLUE, True)
arrow([(1035, 550+D), (1035, 520+D), (505, 520+D), (505, 469+D)], TEAL, True)
text(655, 527+D, "retrieval reads the same index", 10, TEAL)
box(40, 375+D, 170, 94, "Assertions", ["User-written paragraph", "Claims to check", "Example: mission dates"], BLUE)
box(230, 375+D, 170, 94, "Claim extractor", ["ChatDatabricks", "Atomic claims + queries", "AUTHOR_PROMPTS extractor"], BLUE)
box(420, 375+D, 170, 94, "Retriever", ["Claim + entity query", "Rank fusion; 4 passages", "Max 2 per source"], BLUE)
box(610, 375+D, 170, 94, "Verifier + checks", ["ChatDatabricks + Python", "AUTHOR_PROMPTS verifier", "Exact quotes; provenance"], BLUE)
box(800, 375+D, 170, 94, "Quality metrics", ["Citation coverage", "Abstentions and errors", "Potential conflicts"], BLUE)
box(990, 375+D, 170, 94, "Cited report", ["Deterministic HTML", "Claim / verdict / quote", "Links from source data"], BLUE)
for left in (210, 400, 590, 780, 970):
    arrow([(left, 422+D), (left+20, 422+D)], BLUE)

box(515, 220+D, 260, 96, "One bounded retry", ["Only insufficient evidence + a new query", "Search FAISS; append up to 2 passages", "Reverify, then finish regardless of verdict"], BLUE, "#EFF4FF")
arrow([(695, 375+D), (695, 316+D)], BLUE)
text(705, 348+D, "if useful", 10, BLUE)
arrow([(775, 267+D), (885, 267+D), (885, 375+D)], BLUE)
text(812, 279+D, "finish", 10, BLUE)
text(40, 333+D, "State", 11, BLUE, True)
text(40, 314+D, "claims | evidence | verdicts | query history", 11, MUTED)
text(40, 296+D, "retry count | metrics | trace | report", 11, MUTED)
text(40, 263+D, "Input questions need explicit assertions first.", 10.5, MUTED)
text(40, 245+D, "At most 6 claims; at most one follow-up per claim.", 10.5, MUTED)

# Live run on Databricks (left) and the offline prompt-engineering loop (right).
box(40, 218, 330, 108, "Live golden-set run", ["Databricks Free Edition, serverless compute", "LLM: databricks-meta-llama-3-1-8b-instruct", "Embeddings: databricks-bge-large-en", "6/6 golden claims correct; citation coverage 100%"], TEAL, "#EEF8F8")
box(420, 218, 740, 108, "Prompt engineering loop (offline, promptfoo)", [], GOLD, "#FFF9EE")
for i, line in enumerate(["Business goal + labelled evals drive the prompt text", "30 labelled verifier claims (supported / refuted / insufficient)", "8 extractor paragraphs with reference atomic claims", "Same 8B Databricks endpoint and notebook message format"]):
    text(433, 218+108-47-i*15, line, 10.5, MUTED)
for i, line in enumerate(["5 scored tuning rounds; a regressing variant was dropped", "Final verifier 26/30 (87%); final extractor 8/8", "Each round targets an observed failure mode", "Winning prompts are pasted into AUTHOR_PROMPTS"]):
    text(793, 218+108-47-i*15, line, 10.5, MUTED)
arrow([(1160, 272), (1182, 272), (1182, 612), (740, 612), (740, 589)], GOLD, True)
text(905, 618, "AUTHOR_PROMPTS (extractor + verifier)", 10, GOLD)

text(40, 185, "03  DIAGNOSTIC EVALUATION AND INTERPRETATION", 11, TEAL, True)
box(40, 60, 320, 106, "Hand-labelled cases", ["Six candidate claims; labels supplied by author", "Enter the same graph at retrieval (skip extractor)", "Score accuracy and per-class results", "Inspect extraction separately on the demo"], TEAL)
arrow([(360, 113), (392, 113), (392, 470), (505, 470), (505, 495)], TEAL, True)
box(420, 60, 355, 106, "What the checks establish", ["Exact quotes must occur in a retrieved passage.", "Supported/refuted results require cited evidence.", "Invalid output earns no evaluation credit.", "Quote presence does not establish entailment."], TEAL)
box(815, 60, 345, 106, "How to read the results", ["Coverage is enforced; also inspect abstention rate.", "No decided claims means coverage is N/A.", "Conflicting stances flag potential disagreement.", "Shared Wikipedia sources are not independence."], TEAL)
text(40, 25, "Two vector queries = multi-query dense retrieval. Evaluation uses the frozen corpus; it is not run recursively inside the application.", 10, MUTED)
c.showPage()
c.save()
print(OUT)
