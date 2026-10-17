"""Notebook-embeddable workflow; model and search clients are injected for testing.

This module intentionally contains no authored LLM prompts. Supply your own
extractor and verifier system prompts when constructing CrossCheck.
"""

import hashlib
import html
import json
import re
from collections import Counter
from typing import Any, TypedDict
from urllib.parse import urlparse


VERDICTS = {"supported", "refuted", "insufficient-evidence"}
EXTRACT_SCHEMA = {
    "claims": [{"text": "string", "search_query": "string"}],
}
VERIFY_SCHEMA = {
    "verdict": "supported | refuted | insufficient-evidence",
    "citations": [{"passage_id": "string", "quote": "string",
                   "stance": "supports | refutes | context"}],
    "next_query": "string or empty string",
}


class CrossCheckState(TypedDict, total=False):
    text: str
    claims: list[dict[str, Any]]
    evidence: dict[str, list[dict[str, Any]]]
    verdicts: dict[str, dict[str, Any]]
    queries: dict[str, list[str]]
    retry_count: int
    metrics: dict[str, Any]
    trace: list[str]
    report: str


def validate_prompts(prompts):
    for name in ("extractor", "verifier"):
        value = prompts.get(name, "")
        if not isinstance(value, str) or not value.strip() or "TODO" in value:
            raise ValueError(f"Write your own {name} prompt in AUTHOR_PROMPTS first.")


def parse_json_response(response):
    content = response.content
    if isinstance(content, list):
        content = "".join(block.get("text", "") for block in content
                          if isinstance(block, dict) and block.get("type") == "text")
    if not isinstance(content, str):
        raise ValueError("Model response contains no text.")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Model response must be one JSON object.")
    return result


def wikipedia_url(url):
    parsed = urlparse(url)
    return (parsed.scheme == "https" and parsed.hostname == "en.wikipedia.org"
            and parsed.path.startswith("/wiki/") and parsed.username is None)


def passage_from_document(doc):
    source = str(doc.metadata.get("source", ""))
    title = str(doc.metadata.get("title", ""))
    text = doc.page_content
    if not wikipedia_url(source) or not title or not text.strip():
        raise ValueError("Retrieved document is missing valid Wikipedia provenance.")
    passage_id = hashlib.sha256((source + "\n" + text).encode()).hexdigest()[:16]
    return {"passage_id": passage_id, "title": title, "source": source, "text": text}


def merge_ranked_results(result_lists, limit=4, per_source=2):
    """Reciprocal rank fusion, with passage deduplication and source diversity.

    Both lists come from the same dense index: this is multi-query retrieval.
    Keeping two passages per source avoids losing a relevant second chunk.
    """
    passages, scores = {}, {}
    for results in result_lists:
        seen = set()
        for rank, doc in enumerate(results, 1):
            passage = passage_from_document(doc)
            key = passage["passage_id"]
            if key in seen:
                continue
            seen.add(key)
            passages[key] = passage
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
    selected, counts = [], Counter()
    for key in sorted(scores, key=lambda key: (-scores[key], key)):
        passage = passages[key]
        if counts[passage["source"]] < per_source:
            selected.append(passage)
            counts[passage["source"]] += 1
        if len(selected) == limit:
            break
    return selected


def abstention(error="", next_query=""):
    return {"verdict": "insufficient-evidence", "citations": [],
            "conflict": False, "validation_error": error, "next_query": next_query}


def validate_verdict(raw, evidence):
    """Validate quote provenance, not semantic entailment (which needs review)."""
    verdict = raw.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError("invalid_verdict")
    proposed = raw.get("citations")
    if not isinstance(proposed, list) or len(proposed) > 6:
        raise ValueError("invalid_citations")
    next_query = raw.get("next_query", "")
    if not isinstance(next_query, str) or len(next_query) > 300:
        raise ValueError("invalid_next_query")
    by_id = {p["passage_id"]: p for p in evidence}
    citations, seen = [], set()
    for citation in proposed:
        if not isinstance(citation, dict):
            raise ValueError("invalid_citation_object")
        passage = by_id.get(citation.get("passage_id"))
        quote = citation.get("quote")
        stance = citation.get("stance")
        if passage is None:
            raise ValueError("unknown_passage")
        if not isinstance(quote, str) or not quote.strip() or quote not in passage["text"]:
            raise ValueError("quote_not_in_passage")
        if stance not in {"supports", "refutes", "context"}:
            raise ValueError("invalid_stance")
        key = (passage["passage_id"], quote, stance)
        if key not in seen:
            seen.add(key)
            # The model cannot supply or change a source URL or title.
            citations.append({"passage_id": passage["passage_id"], "quote": quote,
                              "source": passage["source"], "title": passage["title"],
                              "stance": stance})
    stances = {c["stance"] for c in citations}
    conflict = {"supports", "refutes"} <= stances
    if conflict:
        verdict = "insufficient-evidence"
    elif verdict == "supported" and "supports" not in stances:
        raise ValueError("missing_supporting_quote")
    elif verdict == "refuted" and "refutes" not in stances:
        raise ValueError("missing_refuting_quote")
    return {"verdict": verdict, "citations": citations, "conflict": conflict,
            "validation_error": "", "next_query": next_query.strip()}


def citation_metrics(verdicts):
    values = list(verdicts.values())
    decided = [v for v in values if v["verdict"] != "insufficient-evidence"]
    count = len(values)
    return {
        "claim_count": count,
        "decided_count": len(decided),
        "citation_coverage": (sum(bool(v["citations"]) for v in decided) / len(decided)
                              if decided else None),
        "abstention_rate": (sum(v["verdict"] == "insufficient-evidence" for v in values)
                            / count if count else None),
        "validation_errors": sum(bool(v["validation_error"]) for v in values),
        "potential_conflicts": sum(v["conflict"] for v in values),
    }


def render_report(claims, verdicts, metrics):
    """Deterministic HTML keeps the final report tied to validated outputs."""
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for claim in claims:
        result = verdicts[claim["id"]]
        citations = "<br><br>".join(
            f'<q>{esc(c["quote"])}</q><br><a href="{esc(c["source"])}">'
            f'{esc(c["title"])}</a> ({esc(c["stance"])})'
            for c in result["citations"])
        notes = []
        if result["conflict"]:
            notes.append("Potential conflicting evidence; human review required.")
        if result["validation_error"]:
            notes.append("Output rejected: " + result["validation_error"])
        if not citations:
            citations = "No qualifying quoted evidence."
        rows.append(f'<tr><td>{esc(claim["text"])}</td><td>{esc(result["verdict"])}'
                    f'</td><td>{citations}</td><td>{esc(" ".join(notes))}</td></tr>')
    counts = Counter(v["verdict"] for v in verdicts.values())
    coverage = metrics["citation_coverage"]
    label = f"{coverage:.0%}" if coverage is not None else "N/A (no decided claims)"
    return (
        '<div style="font-family:Arial,sans-serif;line-height:1.5">'
        '<h2>CrossCheck - Wikipedia claim verification</h2>'
        f'<p>{counts["supported"]} supported; {counts["refuted"]} refuted; '
        f'{counts["insufficient-evidence"]} insufficient evidence. '
        f'Citation coverage of decided claims: {label}.</p>'
        '<table border="1" cellpadding="8" style="border-collapse:collapse;width:100%">'
        '<thead><tr><th>Claim</th><th>Verdict</th><th>Evidence</th><th>Review</th></tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table>'
        '<p>Verdicts are relative to the retrieved corpus. Quote matching verifies '
        'provenance; it does not prove that a quote logically settles the claim. '
        'Wikipedia articles may share sources.</p></div>'
    )


class CrossCheck:
    def __init__(self, extractor, verifier, search, prompts, max_claims=6, k=4):
        validate_prompts(prompts)
        self.extractor = extractor
        self.verifier = verifier
        self.search = search
        self.prompts = prompts
        self.max_claims = max_claims
        self.k = k

    def call(self, model, name, payload):
        # Payload is data and output shape; task instructions are authored by the user.
        response = model.invoke([("system", self.prompts[name]),
                                 ("human", json.dumps(payload, ensure_ascii=False))])
        return parse_json_response(response)

    def extract(self, state):
        raw = self.call(self.extractor, "extractor", {
            "input_text": state["text"], "max_claims": self.max_claims,
            "output_schema": EXTRACT_SCHEMA,
        })
        candidates = raw.get("claims")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= self.max_claims:
            raise ValueError("Extractor must return 1 to max_claims claims.")
        claims = []
        for item in candidates:
            if not isinstance(item, dict):
                raise ValueError("Each extracted claim must be an object.")
            text, query = item.get("text"), item.get("search_query")
            if (not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000
                    or not isinstance(query, str) or not 1 <= len(query.strip()) <= 300):
                raise ValueError("Invalid claim text or search_query.")
            if text.strip() not in [c["text"] for c in claims]:
                claims.append({"id": f"c{len(claims)+1}", "text": text.strip(),
                               "search_query": query.strip()})
        return {"claims": claims, "trace": state.get("trace", []) + ["extract"]}

    def retrieve(self, state):
        evidence, queries = {}, {}
        for claim in state["claims"]:
            terms = list(dict.fromkeys([claim["text"], claim["search_query"]]))
            results = [self.search(term, k=self.k) for term in terms]
            evidence[claim["id"]] = merge_ranked_results(results)
            queries[claim["id"]] = terms
        return {"evidence": evidence, "queries": queries, "retry_count": 0,
                "trace": state.get("trace", []) + ["retrieve"]}

    def verify_one(self, claim, evidence):
        if not evidence:
            return abstention(next_query=claim["search_query"])
        # Transport/auth/quota errors propagate: they are not evidence verdicts.
        response = self.verifier.invoke([
            ("system", self.prompts["verifier"]),
            ("human", json.dumps({"claim": claim["text"], "passages": evidence,
                                  "output_schema": VERIFY_SCHEMA}, ensure_ascii=False)),
        ])
        try:
            return validate_verdict(parse_json_response(response), evidence)
        except (ValueError, TypeError, KeyError) as error:
            return abstention(error=str(error))

    def verify(self, state):
        verdicts = {c["id"]: self.verify_one(c, state["evidence"][c["id"]])
                    for c in state["claims"]}
        return {"verdicts": verdicts, "trace": state.get("trace", []) + ["verify"]}

    def retry_ids(self, state):
        if state.get("retry_count", 0) >= 1:
            return []
        return [key for key, value in state["verdicts"].items()
                if value["verdict"] == "insufficient-evidence" and value["next_query"]
                and value["next_query"].casefold() not in
                [q.casefold() for q in state["queries"][key]]]

    def retry(self, state):
        evidence, verdicts, queries = (dict(state[key]) for key in
                                       ("evidence", "verdicts", "queries"))
        pending = self.retry_ids(state)
        for claim in state["claims"]:
            key = claim["id"]
            if key not in pending:
                continue
            query = verdicts[key]["next_query"]
            new = merge_ranked_results([self.search(query, k=self.k)], limit=2)
            existing_ids = {p["passage_id"] for p in evidence[key]}
            # Preserve first-pass evidence, append at most two new passages.
            evidence[key] = evidence[key] + [p for p in new if p["passage_id"] not in existing_ids]
            queries[key] = queries[key] + [query]
            verdicts[key] = self.verify_one(claim, evidence[key])
        return {"evidence": evidence, "verdicts": verdicts, "queries": queries,
                "retry_count": 1, "trace": state.get("trace", []) + ["retry"]}

    def metrics(self, state):
        return {"metrics": citation_metrics(state["verdicts"]),
                "trace": state.get("trace", []) + ["metrics"]}

    def report(self, state):
        return {"report": render_report(state["claims"], state["verdicts"], state["metrics"]),
                "trace": state.get("trace", []) + ["report"]}

    def build_graph(self):
        from langgraph.graph import StateGraph, START, END

        graph = StateGraph(CrossCheckState)
        for name in ("extract", "retrieve", "verify", "retry", "metrics", "report"):
            graph.add_node(name, getattr(self, name))
        # Golden-set cases enter as pre-labelled atomic claims, bypassing extraction.
        graph.add_conditional_edges(START, lambda s: "retrieve" if s.get("claims") else "extract",
                                    {"extract": "extract", "retrieve": "retrieve"})
        graph.add_edge("extract", "retrieve")
        graph.add_edge("retrieve", "verify")
        graph.add_conditional_edges("verify", lambda s: "retry" if self.retry_ids(s) else "metrics",
                                    {"retry": "retry", "metrics": "metrics"})
        graph.add_edge("retry", "metrics")
        graph.add_edge("metrics", "report")
        graph.add_edge("report", END)
        return graph.compile()


def evaluate_cases(graph, cases):
    """Evaluate retrieval + verification + bounded retry, without extractor noise.

    An invalid response cannot count as a correct insufficient-evidence answer.
    This is a diagnostic set, not an unbiased production benchmark.
    """
    if not cases or any(c.get("expected") not in VERDICTS for c in cases):
        raise ValueError("Hand-label every golden case with an allowed expected verdict.")
    if {c["expected"] for c in cases} != VERDICTS:
        raise ValueError("Include supported, refuted, and insufficient-evidence cases.")
    rows, outputs = [], {}
    for i, case in enumerate(cases, 1):
        claim = {"id": "c1", "text": case["claim"],
                 "search_query": case.get("search_query", case["claim"])}
        result = graph.invoke({"claims": [claim]}, {"recursion_limit": 20})
        verdict = result["verdicts"]["c1"]
        outputs[str(i)] = verdict
        rows.append({"claim": case["claim"], "expected": case["expected"],
                     "actual": verdict["verdict"],
                     "correct": verdict["verdict"] == case["expected"]
                                and not verdict["validation_error"],
                     "validation_error": verdict["validation_error"],
                     "citations": verdict["citations"], "trace": result["trace"]})
    metrics = citation_metrics(outputs)
    metrics["verdict_accuracy"] = sum(bool(r["correct"]) for r in rows) / len(rows)
    metrics["per_class_accuracy"] = {
        label: sum(bool(r["correct"]) for r in rows if r["expected"] == label)
               / sum(r["expected"] == label for r in rows)
        for label in sorted(VERDICTS)
    }
    return {"metrics": metrics, "cases": rows}
