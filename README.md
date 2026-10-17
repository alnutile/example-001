# CrossCheck — Aimpoint working draft

CrossCheck checks a paragraph of historical assertions against a small Apollo /
Space Race Wikipedia corpus. It extracts claims, searches for evidence, verifies
each claim, optionally searches again once, and renders a cited verdict table.

## Start here

1. Import `01_agentic_wikipedia_aimpoint_interview.ipynb` into your Databricks Free
   Edition workspace and attach serverless compute. The notebook is standalone.
2. Run setup and corpus cells. Confirm the configured embedding and chat endpoint
   names exist in your workspace. The default names are inherited from the scaffold,
   not verified against your account.
3. Write the two `AUTHOR_PROMPTS` in your own words. Section 2 describes their input,
   output, and required decisions without supplying finished prompts.
4. Inspect the corpus, then hand-label the six `GOLDEN_CASES`. Include all three
   verdicts and record evidence or a reason in `label_note` before seeing predictions.
5. Run the application and evaluation. Review quote relevance, extraction fidelity,
   abstentions, errors, and the recorded graph path. Adjust based on observed failures.
6. Write Part B yourself. Fresh **Run All**, export HTML, then print to PDF. The
   assessment requests `FIRSTNAME_LASTNAME_AIMPOINT_MLE_ASSIGNMENT.pdf` plus a separate
   reference architecture PDF/PNG. Confirm the deadline in the assessment email.

The first Run All will intentionally stop at the empty author prompts. Evaluation
will also stop until every case has a valid label. This is an implementation draft,
not a completed or executed submission.

## Files

- `01_agentic_wikipedia_aimpoint_interview.ipynb`: primary importable notebook.
- `output/pdf/crosscheck-architecture.pdf`: one-page reference architecture draft.
- `output/crosscheck-architecture.png`: image version of the architecture.
- `output/crosscheck-architecture.mmd`: editable Mermaid diagram.
- `crosscheck/workflow.py`: editable workflow implementation, embedded in the notebook.
- `crosscheck/corpus.py`: corpus loading and JSON snapshot cache, also embedded.
- `tests/`: offline tests using fake models and documents; no credentials required.
- `reference/`: preserved supplied notebook, setup PDF, and design brief.

## Decisions made while implementing the brief

- **Assertion input:** verify claims supplied by the user. A bare topic or question
  needs to be turned into assertions explicitly, avoiding hidden claim invention.
- **Loader fix:** call `WikipediaLoader` with one string at a time. The scaffold
  passes a list even though the loader's interface accepts a string.
- **Small corpus:** start with eight queries, five results each, and a global cap
  of 40 unique articles. Cache text as JSON and record its hash and retrieval time.
- **Passage retrieval:** split article extracts into 1,000-character chunks with
  150 overlap. Merge two dense searches by reciprocal rank fusion. This is query
  expansion; it does not include a lexical/BM25 index. Keep at most two initial
  passages from any one source rather than discarding useful second passages.
- **Bounded decision:** an insufficient verdict with a new query can trigger one
  search and re-verification. The graph cannot retry indefinitely.
- **Quote validation:** source IDs must exist and quotes must be exact substrings.
  URLs and titles come from corpus metadata. A validation failure becomes an
  explicit abstention and cannot score as a correct answer during evaluation.
- **Deterministic report:** render the verified table and counts directly. Two
  author-written prompts are sufficient; no extra summarization model is needed.
- **Separate evaluation run:** golden claims enter the same graph at retrieval.
  Evaluation never recursively runs inside the application. It measures retrieval
  and verification; extraction is checked separately.

These decisions are implemented design choices for you to review, not Part B answers.

## Validation status

- 17 offline tests pass, including real LangGraph execution with fake model clients.
- Notebook schema and Python cells validate; both embedded modules match the source.
- The supplied Part B section is unchanged.
- Direct dependency pins resolve together in a clean local Python 3.12 environment
  using pip's dry run. This is not a Databricks runtime compatibility guarantee.
- Architecture PDF rendered and visually reviewed.
- **Pending:** Databricks imports, endpoint availability, live Wikipedia access,
  actual model outputs/metrics, user prompts, labels, reflection, and final export.

Run local checks:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_notebook.py
```

`scripts/build_notebook.py` is the initial generator and refuses to overwrite an
existing notebook. Keep subsequent author edits in the notebook. If changing a
Python module, replace only its corresponding embedded code cell; the check script
detects divergence. Do not regenerate over prompts or reflection.

## GitHub Actions

The [CrossCheck CI workflow](.github/workflows/ci.yml) runs on every push and pull
request. You can also run it manually from the repository's Actions tab. It uses
Python 3.12 on Ubuntu to install `requirements-local.txt`, run all offline tests,
and validate notebook structure, Python syntax, embedded source consistency, and
the preserved Part B questions. No Databricks credentials or repository secrets
are needed.

After pushing this project to GitHub, look for **CrossCheck CI / Offline tests and
notebook validation** on the commit or pull request. The step logs show individual
test results. A green check establishes that these offline checks passed; live
Databricks access, model quality, and a completed notebook Run All remain separate
validation steps. Blank author prompts and evaluation labels are allowed by CI.

Workflow setup follows GitHub's [Python build and test guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).

## Interpreting evaluation

Citation coverage is enforced for decided claims, so it is a structural metric.
An all-abstention run reports coverage as N/A, not 100%. Quote matching does not
establish entailment. Read accuracy alongside abstention rate and validation errors,
and manually inspect evidence. Contradictory model-assigned stances indicate a
potential conflict, not a proven dispute between sources. Wikipedia articles can
share sources, and short extracts can omit relevant evidence. Six hand-labelled
cases are a development diagnostic, especially after prompt tuning.

## Technical references checked

- [WikipediaLoader source and string query interface](https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/document_loaders/wikipedia.py)
- [ChatDatabricks model invocation and workspace authentication](https://docs.langchain.com/oss/python/integrations/chat/databricks)
- [LangGraph conditional edges](https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_conditional_edges)

The local package versions are recorded in notebook output during a live run.
