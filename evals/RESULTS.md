# CrossCheck prompt tuning with promptfoo

Target: `databricks-meta-llama-3-1-8b-instruct`, temperature 0, max_tokens 2000 (verifier) / 1500 (extractor), same as the notebook.
Messages match the notebook exactly: system = prompt, user = JSON payload (`claim`, `passages`, `output_schema` / `input_text`, `max_claims`=6, `output_schema`).
Grader for extractor llm-rubric: `databricks-meta-llama-3-3-70b-instruct`.

## Datasets
- `verifier_tests.json`: 30 cases (13 supported / 7 refuted / 10 insufficient). Passages are retrieved the way the notebook does it: same corpus (load_corpus, 31 articles, 292 chunks, sha 84bc3395…), same splitter, bge-large-en embeddings, two searches (claim + search_query), k=4, `merge_ranked_results`. Includes the 6 GOLDEN_CASES, a retry case (Apollo 13 with retry passages appended) and a prompt-injection passage case. Labels are based on the retrieved passage text.
- `extractor_tests.json`: 8 paragraphs with reference atomic claims and allowed claim-count ranges (including opinion/question filtering and an injection attempt).

## Assertions
- Verifier (`assert_verifier.py`): parses like `parse_json_response`, then runs the notebook's own `validate_verdict` (schema, known passage_id, quote verbatim in passage, stance consistency, conflict→abstain). Pass = no validation error, verdict == expected, and a non-empty `next_query` when abstaining on cases that need a retry.
- Extractor (`assert_extractor.py` + llm-rubric): JSON parses, 1–6 claims, `text`/`search_query` strings, claim count in expected range, rubric checks atomicity, no added facts, opinions excluded, coverage.

## Scores (pass / total)
| Round | Verifier | Extractor | Notes |
|---|---|---|---|
| 1 | v1 16/30 | v1 2/8 | One-line baselines. Quotes paraphrased; "not mentioned" treated as refuted; extractor wraps JSON in prose. |
| 2 | v2 17/30 | v2 0/8 (1/8 rerun) | Added verdict definitions, verbatim quote rule, JSON-only output. Extractor over-splits to 6 near-duplicate claims. |
| 3 | v3 24/30 (25/30 rerun) | v3 5/8 (6/8 rerun) | Verifier: step-by-step, 1–2 short citations, cite only relevant sentences. Extractor: max_claims is a cap, not a target; no duplicates; keep false user claims. |
| 4 | v4 24/30 | **v4 8/8** | Verifier: same-subject rule ("Apollo 11 first" ≠ "Apollo 13 didn't land"), exactly one quote. Extractor: worked example. Verifier failures were now almost all capitalized first letters / dropped quotation marks. |
| 5 | **v5 26/30 (26/30 rerun)** | – | Verifier: 5–25 word span, "do NOT capitalize the first word", no added/dropped quote marks, quote and passage_id from the same passage. |
| 6 | v6 23/30 | – | Added a final "quote must name the subject" self-check. It made stances worse (refuted with "supports" quotes), so it was dropped. |

**Final: extractor v4 (8/8), verifier v5 (26/30 = 87%, stable across 2 runs).**

Remaining v5 failures: Apollo 13 first-pass (marks refuted from Apollo 11 text with the wrong stance; the code turns this into a validation-error abstention), Apollo 13 after retry and Soviet cosmonauts (quote not verbatim), and Apollo 8 orbit (refuted from wrong-subject text). These are the 8B model's limits: "wrong-subject refutation" and copying text exactly. Worth noting in the reflection: a bigger verifier, or fuzzy quote normalization, would help.

Caveat: the 6 golden cases were part of the tuning set, so their scores are optimistic (the notebook itself warns about this).

## Files
prompts/*.txt (all variants), prompt_fns.py, verifier.yaml, extractor.yaml, run.sh, summarize.py, results/*.json (raw promptfoo outputs), build_index.py, retrieval.py, claims.py, write_nb.py.
