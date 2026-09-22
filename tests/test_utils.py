import pytest
from llm_preclassifier.utils import _flatten_text, _classification_cache_key

def test_flatten_text():
    assert _flatten_text(" Hello WORLD ") == "hello world"
    assert _flatten_text([{"type": "text", "text": "Chunk 1 "}, {"type": "image_url"}]) == "chunk 1"
    assert _flatten_text(None) == ""

def test_classification_cache_key():
    messages1 = [{"role": "user", "content": "Fix it."}]
    messages2 = [{"role": "user", "content": "Fix it."}]
    assert _classification_cache_key(messages1) == _classification_cache_key(messages2)
    
    # Different context alters the key
    messages3 = [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Fix it."}]
    assert _classification_cache_key(messages1) != _classification_cache_key(messages3)
