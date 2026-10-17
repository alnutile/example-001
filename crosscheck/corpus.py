"""Small Wikipedia corpus with a reusable, inspectable JSON snapshot."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def load_corpus(query_terms, max_docs=40, per_query=5, chars_per_article=6000,
                cache_dir="/tmp/crosscheck-cache", refresh=False):
    from langchain_community.document_loaders import WikipediaLoader
    from langchain_core.documents import Document

    if not query_terms or any(not isinstance(q, str) or not q.strip() for q in query_terms):
        raise ValueError("query_terms must contain nonempty strings.")
    if max_docs < 1 or per_query < 1:
        raise ValueError("Document limits must be positive.")
    config = {"query_terms": query_terms, "max_docs": max_docs, "per_query": per_query,
              "chars_per_article": chars_per_article, "language": "en", "version": 1}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"wikipedia-{key}.json"
    if path.exists() and not refresh:
        snapshot = json.loads(path.read_text())
        if snapshot["config"] != config:
            raise ValueError("Cached corpus configuration does not match.")
        digest = hashlib.sha256(json.dumps(snapshot["documents"], sort_keys=True).encode()).hexdigest()
        if digest != snapshot["manifest"]["corpus_sha256"]:
            raise ValueError("Cached corpus content changed; refresh the snapshot explicitly.")
        docs = [Document(page_content=d["text"], metadata=d["metadata"])
                for d in snapshot["documents"]]
        return docs, {**snapshot["manifest"], "cache_hit": True, "cache_path": str(path)}

    docs, seen, empty_queries = [], set(), []
    for query in query_terms:
        if len(docs) >= max_docs:
            break
        # WikipediaLoader takes a single string, not the scaffold's list.
        loaded = WikipediaLoader(query=query, lang="en", load_max_docs=per_query,
                                 doc_content_chars_max=chars_per_article).load()
        if not loaded:
            empty_queries.append(query)
        for doc in loaded:
            source = doc.metadata.get("source")
            if source and source not in seen and doc.page_content.strip():
                seen.add(source)
                docs.append(doc)
                if len(docs) == max_docs:
                    break
    if not docs:
        raise RuntimeError("Wikipedia returned no documents; check network access and queries.")
    records = [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    manifest = {"document_count": len(docs), "corpus_sha256": digest,
                "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "empty_queries": empty_queries}
    snapshot = {"config": config, "manifest": manifest, "documents": records}
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    temp.replace(path)
    return docs, {**manifest, "cache_hit": False, "cache_path": str(path)}
