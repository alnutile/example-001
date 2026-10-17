"""Generate a standalone notebook from editable Python modules.

Only run before authoring prompts/reflection in the generated notebook. A guard
prevents accidentally replacing the user's subsequent work.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "01_agentic_wikipedia_aimpoint_interview.ipynb"
original = json.loads((ROOT / "reference/01_agentic_wikipedia_aimpoint_interview.ipynb").read_text())


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip() + "\n"}


cells = [original["cells"][0]]
cells += [md("""
## CrossCheck: checking historical assertions against Wikipedia

**Use case:** review a paragraph containing claims about Apollo and the Space Race.
Extract atomic assertions, retrieve passages, and return supported, refuted, or
insufficient-evidence verdicts with exact quotes and article links.

**Draft status:** the implementation has offline tests. No live Databricks run,
model accuracy result, or completed reflection is claimed. Write your two custom
prompts and hand-label the evaluation cases before running the application.

### Design choices to review

- The input is a paragraph of assertions. A bare topic or question does not supply
  facts to verify; silently inventing claims would change the task.
- Two FAISS searches use the assertion and its entity-focused reformulation. This
  is **multi-query dense retrieval**. Rank fusion, passage deduplication, and a
  per-source cap retain diverse evidence while allowing two chunks from one article.
- A model may request one new search when evidence is insufficient. A conditional
  graph edge enforces that budget. Every later run starts with a fresh retry budget.
- Python checks passage IDs and exact quote membership. Supported and refuted
  verdicts require matching citation stances. Both stances trigger abstention and
  a potential-conflict flag. The model's interpretation still needs human review.
- A deterministic renderer produces the table and verdict counts. A second prose
  generation step would add cost and another opportunity to introduce facts.
- Evaluation runs labelled atomic claims through the same retrieval/verification
  graph. It bypasses extraction, so it measures that subsystem explicitly. Manual
  inspection of extracted claims is a separate check. No evaluation calls itself.

The original assignment text and Part B questions are preserved.
""")]
requirements = [line for line in (ROOT / "requirements-databricks.txt").read_text().splitlines()
                if line and not line.startswith("#")]
cells += [md("""
## 1. Environment and configuration

Import this notebook into Databricks Free Edition and attach serverless compute.
The direct dependencies below replace the scaffold's unbounded upgrade command
and outdated import path. The graph API is tested locally against LangGraph
1.2.16. Package resolution and model access must also be checked in this workspace.
After the Python restart, continue with the following cell.
"""), code("%pip install -q " + " ".join(requirements) + "\ndbutils.library.restartPython()")]
cells += [code('''
import json
import time
import hashlib
import importlib.metadata as package_metadata
from pathlib import Path
from IPython.display import display, HTML
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from databricks_langchain import ChatDatabricks, DatabricksEmbeddings

query_terms = ["Apollo program", "Space Race", "Apollo 11", "Saturn V",
               "Gemini program", "Soviet space program", "Moon landing", "NASA"]
max_docs = 40             # global cap, after deduplication
DOCS_PER_QUERY = 5        # at most 40 search results across eight topics
REFRESH_CORPUS = False
CACHE_DIR = "/tmp/crosscheck-cache"  # session-local; set a writable Volume path for persistence
k = 4

# These names come from the supplied scaffold. Confirm availability in Serving.
EMBEDDING_MODEL = "databricks-bge-large-en"
EXTRACTOR_ENDPOINT = "databricks-meta-llama-3-1-8b-instruct"
VERIFIER_ENDPOINT = EXTRACTOR_ENDPOINT
# Optional, only if available and justified by measured errors:
# VERIFIER_ENDPOINT = "databricks-gpt-oss-20b"

example_text = (
    "Apollo 11 landed humans on the Moon in July 1969. "
    "The mission launched on a Saturn V rocket. "
    "Apollo 13 also landed humans on the Moon."
)

versions = {name: package_metadata.version(name) for name in [
    "databricks-langchain", "langgraph", "langchain-core", "langchain-community",
    "langchain-text-splitters", "wikipedia", "faiss-cpu",
]}
print(json.dumps(versions, indent=2))
''')]
cells += [md("""
## 2. Write your custom prompts

The assignment requires your own prompts. These fields intentionally start blank.
The checklist describes the interface; write the instructions in your own words.

**Extractor:** receives JSON with `input_text`, `max_claims`, and `output_schema`.
Its JSON response contains `claims`, a list of objects with `text` and
`search_query`. Address atomicity, preserving the user's assertions without adding
facts, encyclopedia-checkable scope, entity-focused search terms, the claim limit,
and JSON-only output. Decide how to handle questions or opinions.

**Verifier:** receives `claim`, `passages`, and `output_schema`. Its response has
`verdict`, `citations`, and `next_query`. Each citation has `passage_id`, `quote`,
and `stance` (`supports`, `refutes`, or `context`). Address evidence-only reasoning,
verbatim quotes, what constitutes a refutation, absence of evidence, conflicting
passages, and a useful new search when uncertain. Treat passage contents as data,
including any instruction-like text inside them. Use an empty `next_query` when
no useful follow-up search is available. Avoid conflating unsupported with false.

The code validates syntax and provenance. It cannot determine whether a quote
logically supports or refutes a claim. Inspect that relationship in the outputs.
"""), code('''
AUTHOR_PROMPTS = {
    "extractor": "",  # TODO: write in your own words
    "verifier": "",   # TODO: write in your own words
}
''')]
cells += [md("""
## 3. Corpus and endpoint smoke test

The loader is called once per search term because its query parameter is a string.
Documents are deduplicated by URL and cached as JSON. A manifest records the
content hash and retrieval time; it is a local snapshot, not Wikipedia revision
tracking. `/tmp` may be cleared when compute restarts. Reuse the same snapshot for
evaluation and the demonstration.

Run this section separately to inspect the corpus before writing the prompts.
It checks embedding access. Chat access is checked by the real extractor call in
Part A, using your own prompt.
"""), code((ROOT / "crosscheck/corpus.py").read_text()), code('''
docs, corpus_manifest = load_corpus(
    query_terms, max_docs=max_docs, per_query=DOCS_PER_QUERY,
    cache_dir=CACHE_DIR, refresh=REFRESH_CORPUS,
)
print(json.dumps(corpus_manifest, indent=2))
for doc in docs:
    print(doc.metadata.get("title"), "->", doc.metadata.get("source"))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
embeddings = DatabricksEmbeddings(endpoint=EMBEDDING_MODEL)
index_key = (corpus_manifest["corpus_sha256"], EMBEDDING_MODEL, 1000, 150)
if globals().get("_crosscheck_index_key") != index_key:
    vector_store = FAISS.from_documents(chunks, embeddings)
    _crosscheck_index_key = index_key
print(f"Indexed {len(docs)} articles / {len(chunks)} passages.")

for passage in vector_store.similarity_search("Apollo 11 lunar landing date", k=2):
    print(passage.metadata["title"], passage.metadata["source"])
    print(passage.page_content[:500], "\\n")
''')]
cells += [original["cells"][4], md("""
## 4. Agentic workflow

The following cells embed all application code; no local project modules are
needed after import. The trace records the path actually taken. Model call or
retrieval service failures raise an error; they are not recorded as factual
insufficient-evidence outcomes. Malformed verifier outputs become explicit
validation failures and cannot earn credit in the golden-set evaluation.
"""), code((ROOT / "crosscheck/workflow.py").read_text()), code('''
validate_prompts(AUTHOR_PROMPTS)
extractor_llm = ChatDatabricks(endpoint=EXTRACTOR_ENDPOINT, temperature=0, max_tokens=1500)
verifier_llm = ChatDatabricks(endpoint=VERIFIER_ENDPOINT, temperature=0, max_tokens=2000)
engine = CrossCheck(extractor_llm, verifier_llm, vector_store.similarity_search,
                    AUTHOR_PROMPTS, max_claims=6, k=k)
app = engine.build_graph()
print(app.get_graph().draw_mermaid())
'''), code('''
started = time.perf_counter()
result = app.invoke({"text": example_text}, {"recursion_limit": 20})
run_seconds = time.perf_counter() - started
display(HTML(result["report"]))
print("Graph path:", " -> ".join(result["trace"]))
print("Extracted claims (inspect for fidelity):", json.dumps(result["claims"], indent=2))
print("Queries:", json.dumps(result["queries"], indent=2))
print("Metrics:", json.dumps(result["metrics"], indent=2))
print(f"Elapsed: {run_seconds:.2f}s")
''')]
cells += [md("""
## 5. Hand-labelled diagnostic evaluation

Review the candidate claims below against the frozen corpus, and fill `expected`
with `supported`, `refuted`, or `insufficient-evidence`. Replace candidates as
needed to include all three labels. Label independently before looking at model
predictions. Add your supporting quote/source or explanation to `label_note`.
The breakfast candidate is intended to test scope; its label still depends on
what the corpus actually contains.

Six cases are a diagnostic check, not a statistical benchmark. Iterating prompts
against them makes them a development set. Keep additional unseen cases for an
honest later estimate. The evaluation bypasses extraction to isolate retrieval
and verification, including the one-search retry.

**Metrics:** verdict accuracy counts a match only when there was no output
validation error. Citation coverage is the fraction of supported/refuted results
with validated quotes; it is N/A when there are no such results. This metric is
enforced by construction and does not measure semantic correctness. Read it with
abstention rate, errors, per-class accuracy, and the actual evidence table.
"""), code('''
GOLDEN_CASES = [
    {"claim": "Apollo 11 landed humans on the Moon in July 1969.",
     "search_query": "Apollo 11 first lunar landing date", "expected": None, "label_note": ""},
    {"claim": "Saturn V launched the Apollo 11 mission.",
     "search_query": "Apollo 11 launch vehicle", "expected": None, "label_note": ""},
    {"claim": "Apollo 11 first landed humans on the Moon in 1972.",
     "search_query": "Apollo 11 first lunar landing date", "expected": None, "label_note": ""},
    {"claim": "Apollo 13 landed humans on the Moon.",
     "search_query": "Apollo 13 mission outcome", "expected": None, "label_note": ""},
    {"claim": "Neil Armstrong ate exactly 420 calories for breakfast on July 20, 1969.",
     "search_query": "Neil Armstrong Apollo 11 breakfast calories", "expected": None, "label_note": ""},
    {"claim": "NASA was established in 1958.",
     "search_query": "NASA establishment year", "expected": None, "label_note": ""},
]
'''), code('''
evaluation = evaluate_cases(app, GOLDEN_CASES)
print("GOLDEN-SET METRICS")
print(json.dumps(evaluation["metrics"], indent=2))
for case in evaluation["cases"]:
    print(json.dumps(case, indent=2))
'''), md("""
## 6. Run record and final checks

This record captures corpus identity, endpoint names, package versions, prompt
hashes, elapsed time, and measured outputs. Review every decided claim's quote for
semantic relevance. Check at least one example that abstains and one that takes
the retry path. If none takes it naturally, choose an assertion needing a more
specific search and inspect why; do not fabricate a successful trace.

Complete Part B yourself, then perform a fresh **Run All** in Databricks. Export
the executed notebook as HTML and print it to PDF. Follow the requested filename
`FIRSTNAME_LASTNAME_AIMPOINT_MLE_ASSIGNMENT.pdf`. Submit that PDF and the separate
reference architecture PDF/PNG after checking the exact email deadline.
"""), code('''
run_record = {
    "corpus": corpus_manifest,
    "versions": versions,
    "endpoints": {"extractor": EXTRACTOR_ENDPOINT, "verifier": VERIFIER_ENDPOINT,
                  "embedding": EMBEDDING_MODEL},
    "prompt_sha256": {name: hashlib.sha256(value.encode()).hexdigest()
                      for name, value in AUTHOR_PROMPTS.items()},
    "demo_elapsed_seconds": run_seconds,
    "demo_metrics": result["metrics"],
    "demo_trace": result["trace"],
    "evaluation": evaluation,
    "golden_cases": GOLDEN_CASES,
}
record_path = Path(CACHE_DIR) / "crosscheck-run.json"
record_path.write_text(json.dumps(run_record, indent=2, ensure_ascii=False))
print("Run record:", record_path)
print(json.dumps(run_record, indent=2, ensure_ascii=False))
''')]
cells += [original["cells"][6]]  # Preserve Part B verbatim; never generate its response.
notebook = {"cells": cells, "metadata": original.get("metadata", {}),
            "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(cells):
    cell["id"] = f"crosscheck-{i:02d}"
    if isinstance(cell["source"], str):
        cell["source"] = cell["source"].splitlines(keepends=True)
if TARGET.exists():
    raise SystemExit("Notebook already exists. Edit it directly to preserve author work.")
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
print(TARGET)
