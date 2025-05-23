import pytest

# Chunker
from src.agents.chunker import chunk_text
# Embedder
import src.agents.embedder as embedder_mod
# Planner
from src.agents.planner import plan
# Executor
import src.agents.executor as executor_mod
# Reasoner
import src.agents.reasoner as reasoner_mod



def test_chunk_text_respects_size_and_overlap():
    # With size=4 and overlap=2, text should be split into ['abcd','cdef','ef'] (final remainder chunk included)
    text = "abcdef"
    chunks = chunk_text(text, size=4, overlap=2)
    # final chunk may be shorter than size
    assert chunks == ["abcd", "cdef", "ef"]

@pytest.mark.parametrize("docs,expected", [
    (["a", "b"], [[1, 2], [3, 4]]),
    ([], [])
])
def test_embed_text_consistency_with_embed_texts(monkeypatch, docs, expected):
    
    # Prevent lazy load from actually loading model
    monkeypatch.setattr(embedder_mod, '_lazy_load', lambda: None)
    # Stub internal tokenizer and model so original embed_texts won't run
    monkeypatch.setattr(embedder_mod, '_tokenizer', None)
    monkeypatch.setattr(embedder_mod, '_model', None)
    # Stub embed_texts on the module
    monkeypatch.setattr(embedder_mod, 'embed_texts', lambda texts: expected)

    # embed_text should return the first vector when docs exist
    if docs:
        assert embedder_mod.embed_text(docs[0]) == expected[0]
    # embed_texts returns full list
    assert embedder_mod.embed_texts(docs) == expected




def test_plan_returns_reasonable_subqueries():
    query = "How do antibiotics affect biofilms and how does resistance occur?"
    subqueries = plan(query)

    assert isinstance(subqueries, list)
    assert all(isinstance(q, str) and q.strip() for q in subqueries)
    assert any("biofilm" in q.lower() or "resistance" in q.lower() for q in subqueries)


def test_reason_returns_contextual_answer(monkeypatch):
    import torch
    # Prevent actual model load
    monkeypatch.setattr(reasoner_mod, '_lazy_load_llm', lambda: None)
    # Stub the LLM so .generate(...) returns a fake token list
    reasoner_mod._llm = type("M", (), {
        'device': 'cpu',
        'generate': lambda self, **kwargs: [[0, 1, 2]]
    })()

    # Dummy tokenizer: callable, has decode(), eos_token_id
    class DummyEncoded(dict):
        def __init__(self):
            super().__init__()
            self['input_ids'] = torch.tensor([[0, 1, 2]])
            self['attention_mask'] = torch.tensor([[1, 1, 1]])
        def to(self, device):
            return self

    class DummyTokenizer:
        eos_token_id = 0
        def __call__(self, prompt, return_tensors, truncation, padding):
            return DummyEncoded()
        def decode(self, token_ids, skip_special_tokens=True):
            return "Some prompt Answer: THE_ANS ---"

    # Monkey-patch tokenizer
    reasoner_mod._tokenizer_llm = DummyTokenizer()

    # Call reason() and assert
    ans = reasoner_mod.reason("ignored question", ["ctx1"])
    assert ans == "THE_ANS"




def test_execute_orchestrates_agents(monkeypatch):
    monkeypatch.setattr(executor_mod, 'plan', lambda q: ['s1', 's2'])
    monkeypatch.setattr(executor_mod, 'init_es_client', lambda: None)
    monkeypatch.setattr(executor_mod, 'retrieve', lambda sq, es, top_k: [{'text': sq + '_ctx'}])
    monkeypatch.setattr(executor_mod, 'reason', lambda q, ctx: 'ANS:' + '|'.join(ctx))
    result = executor_mod.execute("input")
    assert result == "ANS:s1_ctx|s2_ctx"
