import pytest
import importlib

# Sectioner
from src.services.sectioner import split_into_sections
# Docs loader
from src.services.docs_loader import load_documents
# Indexer
from src.services.indexer import index_documents
# Retriever
from src.services.retriever import retrieve
# Delete index
from src.services.delete_index import delete_index

# 1) sectioner.py → split_into_sections
def test_split_into_sections_detects_and_splits_headings():
    text = "1 Introduction\nHello\n2 Methods\nWorld"
    sections = split_into_sections(text)
    # Expect two named sections with correct bodies
    assert sections == [
        ("INTRODUCTION", "Hello"),
        ("METHODS", "World")
    ]

# 2) docs_loader.py → load_documents
def test_load_documents_creates_entries_per_section(tmp_path):
    d = tmp_path / "docs"; d.mkdir()
    (d / "a.txt").write_text("Foo bar")
    (d / "b.txt").write_text("INTRODUCTION\nBaz")
    docs = load_documents(str(d))
    types = sorted(doc["section"] for doc in docs)
    assert types == ["FULL_TEXT", "INTRODUCTION"]

# 3) indexer.py → index_documents
def test_index_documents_uses_bulk_and_chunks(monkeypatch):
    # dummy ES client
    class DummyEs:
        def __init__(self):
            self.indices = type("X", (), {"exists": lambda s, i: True})()
        def index(self, **kw): pass
    es = DummyEs()
    # stub dependencies on src.services.indexer
    monkeypatch.setattr('src.services.indexer.load_documents', lambda folder: [{"file_path":"f","section":"S","text":"abcd"}])
    monkeypatch.setattr('src.services.indexer.chunk_text', lambda t: ["ab","cd"] )
    monkeypatch.setattr('src.services.indexer.embed_texts', lambda ch: [[0.1],[0.2]])
    called = {"bulk": False}
    def fake_bulk(es_arg, actions, stats_only):
        called["bulk"] = True
        return (len(actions), 0)
    monkeypatch.setattr('src.services.indexer.bulk', fake_bulk)
    index_documents(es)
    assert called["bulk"] is True

# 4) retriever.py → retrieve
def test_retrieve_parses_es_response(monkeypatch):
    fake_resp = {"hits": {"hits":[{"_score":5.5,"_source":{"file_path":"x","section":"Y","chunk_id":1,"text":"T"}}]}}
    # stub embed_text in src.services.retriever
    monkeypatch.setattr('src.services.retriever.embed_text', lambda q: [1.0,2.0])
    class E:
        def search(self, index, body): return fake_resp
    results = retrieve("q", E(), top_k=1)
    assert results == [{"file_path":"x","section":"Y","chunk_id":1,"text":"T","score":5.5}]


def test_delete_index_prints_when_index_missing(monkeypatch, capsys):
    import importlib
    mod = importlib.import_module("src.services.delete_index")

    # Dummy indices that accept keyword index:
    class DummyIdx:
        def exists(self, *args, **kwargs):
            return False
        def delete(self, index=None, *args, **kwargs):
            # We won't get here in this test
            raise RuntimeError("should not delete")

    class DummyEs:
        def __init__(self, *a, **k):
            self.indices = DummyIdx()

    # Patch the Elasticsearch constructor in the module
    monkeypatch.setattr(mod, "Elasticsearch", DummyEs)

    # Call the function under test
    mod.delete_index()

    # Capture and assert the output
    out = capsys.readouterr().out
    assert f"Index '{mod.INDEX_NAME}' does not exist." in out


