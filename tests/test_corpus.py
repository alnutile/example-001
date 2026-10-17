import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from crosscheck.corpus import load_corpus


class Document:
    def __init__(self, page_content, metadata):
        self.page_content, self.metadata = page_content, metadata


class CorpusTests(unittest.TestCase):
    def test_string_queries_deduplication_and_cache(self):
        calls = []
        class Loader:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.query = kwargs["query"]
            def load(self):
                return [Document("Shared", {"source": "https://en.wikipedia.org/wiki/Common"}),
                        Document(self.query, {"source": f"https://en.wikipedia.org/wiki/{self.query}"})]
        modules = {
            "langchain_community.document_loaders": SimpleNamespace(WikipediaLoader=Loader),
            "langchain_core.documents": SimpleNamespace(Document=Document),
        }
        with patch.dict("sys.modules", modules), tempfile.TemporaryDirectory() as directory:
            docs, manifest = load_corpus(["Apollo", "NASA"], max_docs=3, cache_dir=directory)
            self.assertEqual([c["query"] for c in calls], ["Apollo", "NASA"])
            self.assertEqual(len(docs), 3)
            self.assertFalse(manifest["cache_hit"])
            cached, second = load_corpus(["Apollo", "NASA"], max_docs=3, cache_dir=directory)
            self.assertEqual(len(calls), 2)
            self.assertTrue(second["cache_hit"])
            self.assertEqual(len(cached), 3)
            self.assertEqual(manifest["corpus_sha256"], second["corpus_sha256"])

    def test_changed_cache_is_detected(self):
        class Loader:
            def __init__(self, **kwargs):
                pass
            def load(self):
                return [Document("Original", {"source": "https://en.wikipedia.org/wiki/Apollo"})]
        modules = {
            "langchain_community.document_loaders": SimpleNamespace(WikipediaLoader=Loader),
            "langchain_core.documents": SimpleNamespace(Document=Document),
        }
        with patch.dict("sys.modules", modules), tempfile.TemporaryDirectory() as directory:
            _, manifest = load_corpus(["Apollo"], cache_dir=directory)
            path = manifest["cache_path"]
            with open(path) as handle:
                cached = json.load(handle)
            cached["documents"][0]["text"] = "Changed"
            with open(path, "w") as handle:
                json.dump(cached, handle)
            with self.assertRaisesRegex(ValueError, "content changed"):
                load_corpus(["Apollo"], cache_dir=directory)


if __name__ == "__main__":
    unittest.main()
