import json
import unittest
from types import SimpleNamespace

from crosscheck.workflow import (
    CrossCheck, abstention, citation_metrics, evaluate_cases, merge_ranked_results,
    passage_from_document, render_report, validate_verdict,
)


TEXT = "Apollo 11 landed on the Moon in July 1969. The mission used a Saturn V rocket."
URL = "https://en.wikipedia.org/wiki/Apollo_11"
DOC = SimpleNamespace(page_content=TEXT, metadata={"source": URL, "title": "Apollo 11"})
PASSAGE = passage_from_document(DOC)
PROMPTS = {"extractor": "test fixture", "verifier": "test fixture"}


class FakeModel:
    def __init__(self, respond):
        self.respond = respond
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        result = self.respond(json.loads(messages[-1][1]))
        return SimpleNamespace(content=json.dumps(result))


def cited(payload, verdict="supported", quote=TEXT):
    return {"verdict": verdict, "citations": [{
        "passage_id": payload["passages"][0]["passage_id"], "quote": quote,
        "stance": "supports" if verdict == "supported" else "refutes",
    }], "next_query": ""}


def workflow(verifier, search=None):
    extractor = FakeModel(lambda _: {"claims": [
        {"text": "Apollo 11 landed in 1969.", "search_query": "Apollo 11 landing date"}]})
    return CrossCheck(extractor, verifier, search or (lambda term, k: [DOC]), PROMPTS)


class WorkflowTests(unittest.TestCase):
    def test_graph_happy_path_and_real_provenance(self):
        engine = workflow(FakeModel(cited))
        result = engine.build_graph().invoke({"text": "Apollo 11 landed in 1969."})
        self.assertEqual(result["trace"], ["extract", "retrieve", "verify", "metrics", "report"])
        self.assertEqual(result["verdicts"]["c1"]["citations"][0]["source"], URL)
        self.assertEqual(result["metrics"]["citation_coverage"], 1)

    def test_rejects_invented_quote(self):
        engine = workflow(FakeModel(lambda p: cited(p, quote="It landed in 1972.")))
        result = engine.build_graph().invoke({"text": "Some assertion"})
        verdict = result["verdicts"]["c1"]
        self.assertEqual(verdict["verdict"], "insufficient-evidence")
        self.assertEqual(verdict["validation_error"], "quote_not_in_passage")
        self.assertIsNone(result["metrics"]["citation_coverage"])

    def test_rejects_unknown_passage(self):
        raw = cited({"passages": [PASSAGE]})
        raw["citations"][0]["passage_id"] = "invented"
        with self.assertRaisesRegex(ValueError, "unknown_passage"):
            validate_verdict(raw, [PASSAGE])

    def test_supported_claim_requires_quote(self):
        with self.assertRaisesRegex(ValueError, "missing_supporting_quote"):
            validate_verdict({"verdict": "supported", "citations": []}, [PASSAGE])

    def test_source_url_cannot_be_changed_by_model(self):
        raw = cited({"passages": [PASSAGE]})
        raw["citations"][0]["source"] = "https://example.com"
        result = validate_verdict(raw, [PASSAGE])
        self.assertEqual(result["citations"][0]["source"], URL)

    def test_conflicting_stances_force_abstention(self):
        raw = cited({"passages": [PASSAGE]})
        raw["citations"].append({**raw["citations"][0], "stance": "refutes"})
        result = validate_verdict(raw, [PASSAGE])
        self.assertTrue(result["conflict"])
        self.assertEqual(result["verdict"], "insufficient-evidence")

    def test_retry_is_bounded_even_if_model_keeps_requesting_it(self):
        queries = []
        model = FakeModel(lambda p: {"verdict": "insufficient-evidence", "citations": [],
                                     "next_query": f"new query {len(queries)}"})
        def search(term, k):
            queries.append(term)
            return [DOC]
        result = workflow(model, search).build_graph().invoke({"text": "Assertion"})
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(queries), 3)
        self.assertEqual(result["trace"].count("retry"), 1)

    def test_retry_can_resolve_initial_abstention(self):
        calls = []
        extra = SimpleNamespace(page_content="The launch occurred on July 16, 1969.",
                                metadata={"source": URL, "title": "Apollo 11"})
        def respond(payload):
            calls.append(payload)
            if len(calls) == 1:
                return {"verdict": "insufficient-evidence", "citations": [],
                        "next_query": "Apollo 11 launch date"}
            passage = next(p for p in payload["passages"] if "launch occurred" in p["text"])
            return cited({"passages": [passage]}, quote=passage["text"])
        engine = workflow(FakeModel(respond), lambda term, k: [extra] if "launch" in term else [DOC])
        result = engine.build_graph().invoke({"text": "Assertion"})
        self.assertEqual(result["verdicts"]["c1"]["verdict"], "supported")
        self.assertEqual(len(result["evidence"]["c1"]), 2)

    def test_infrastructure_error_propagates(self):
        class BrokenModel:
            def invoke(self, messages):
                raise RuntimeError("quota exhausted")
        with self.assertRaisesRegex(RuntimeError, "quota exhausted"):
            workflow(BrokenModel()).build_graph().invoke({"text": "Assertion"})

    def test_duplicate_results_do_not_duplicate_passages(self):
        merged = merge_ranked_results([[DOC, DOC], [DOC]])
        self.assertEqual(len(merged), 1)

    def test_two_passages_from_same_source_are_retained(self):
        doc2 = SimpleNamespace(page_content="Second distinct passage.", metadata=DOC.metadata)
        doc3 = SimpleNamespace(page_content="Third distinct passage.", metadata=DOC.metadata)
        self.assertEqual(len(merge_ranked_results([[DOC, doc2, doc3]])), 2)

    def test_empty_corpus_abstains_without_model_call(self):
        model = FakeModel(cited)
        result = workflow(model, lambda term, k: []).build_graph().invoke({"text": "Assertion"})
        self.assertEqual(model.calls, 0)
        self.assertEqual(result["metrics"]["abstention_rate"], 1)

    def test_report_escapes_user_and_source_text(self):
        report = render_report([{"id": "c1", "text": "<script>alert(1)</script>"}],
                               {"c1": abstention()}, citation_metrics({"c1": abstention()}))
        self.assertNotIn("<script>", report)
        self.assertIn("&lt;script&gt;", report)

    def test_eval_bypasses_extractor_and_does_not_reward_invalid_abstention(self):
        engine = workflow(FakeModel(lambda p: cited(p, quote="Invented evidence")))
        cases = [{"claim": label, "expected": label}
                 for label in ["supported", "refuted", "insufficient-evidence"]]
        result = evaluate_cases(engine.build_graph(), cases)
        self.assertEqual(engine.extractor.calls, 0)
        self.assertEqual(result["metrics"]["verdict_accuracy"], 0)
        self.assertEqual(result["metrics"]["validation_errors"], 3)

    def test_labels_must_be_supplied_before_evaluation(self):
        with self.assertRaisesRegex(ValueError, "Hand-label"):
            evaluate_cases(None, [{"claim": "An assertion", "expected": None}])


if __name__ == "__main__":
    unittest.main()
