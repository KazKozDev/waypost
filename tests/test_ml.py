"""Checks of the "Further" new modules: the L1 head, bandit, PII, verifier,
the semantic cache."""
import os
import tempfile
import time

import numpy as np
import pytest

from waypost.bandit import Bandit
from waypost.cache import SemanticCache, _exact_guard, _exact_sequence, _exact_tokens
from waypost.head import TaskHead
from waypost.pii import detect, has_pii
from waypost.verify import Verifier
from waypost.schemas import ChatMessage, ChatRequest


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "t.db")


# ------------------------------------------------------------------ head


def test_head_trains_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y_task = ["code"] * 20 + ["chat"] * 20 + ["extraction"] * 20
    y_comp = [0.8] * 20 + [0.2] * 20 + [0.4] * 20

    head = TaskHead(dim=8)
    head.train(X, y_task, y_comp, epochs=300)
    probs, comp, conf = head.predict(X[0])
    assert max(probs, key=probs.get) == "code"
    assert 0.0 <= comp <= 1.0
    assert 0.0 <= conf <= 1.0


def test_head_roundtrip(db):
    head = TaskHead(dim=4)
    head.W = np.ones((5, 4))
    path = os.path.join(db, "head.json")
    head.save(path)
    loaded = TaskHead.load(path)
    assert loaded.classes == head.classes
    assert loaded.dim == 4
    np.testing.assert_allclose(loaded.W, head.W)


# ----------------------------------------------------------------- bandit


def test_bandit_returns_prior_without_data():
    b = Bandit()
    assert b.quality("code", "p/m", prior=0.7) == 0.7


def test_bandit_learns_from_rewards():
    b = Bandit()
    for _ in range(50):
        b.update("code", "good/m", 1.0)
        b.update("code", "bad/m", 0.0)
    assert b.quality("code", "good/m") > 0.9
    assert b.quality("code", "bad/m") < 0.1


def test_bandit_survives_restart(db):
    b = Bandit(db)
    b.update("code", "p/m", 1.0)
    revived = Bandit(db)
    assert revived.quality("code", "p/m") > 0.5


# ------------------------------------------------------------------- pii


def test_pii_detects_email_and_phone():
    assert "email" in detect("свяжитесь: a.b@example.com")
    assert "phone" in detect("позвоните +7 912 345-67-89")


def test_pii_detects_valid_card_via_luhn():
    assert "card" in detect("карта 4539 1488 0343 6467")  # valid Luhn


def test_pii_ignores_plain_numbers():
    assert detect("сколько будет 6*7 = 42") == set()


def test_pii_detects_passport_keyword():
    assert "passport" in detect("паспорт серия 45 12 номер 345678")


def test_has_pii():
    assert has_pii("мой email a@b.com")
    assert not has_pii("привет, как дела")


# -------------------------------------------------------------- verifier


def _body(content="ответ", finish="stop"):
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content},
            }
        ]
    }


def test_verifier_passes_good_response():
    v = Verifier()
    req = ChatRequest(messages=[ChatMessage(role="user", content="x")])
    assert v.verify(req, None, _body()) == (True, "")


def test_verifier_rejects_empty():
    v = Verifier()
    req = ChatRequest(messages=[ChatMessage(role="user", content="x")])
    ok, reason = v.verify(req, None, _body(""))
    assert not ok and reason == "empty"


def test_verifier_rejects_truncated():
    v = Verifier()
    req = ChatRequest(messages=[ChatMessage(role="user", content="x")])
    ok, reason = v.verify(req, None, _body(finish="length"))
    assert not ok and reason == "truncated"


def test_verifier_checks_json_schema():
    v = Verifier()
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="x")],
        response_format={"type": "json_object"},
    )
    ok, reason = v.verify(req, None, _body("не json"))
    assert not ok and reason == "invalid_json"
    assert v.verify(req, None, _body('{"ok": true}'))[0]


# ------------------------------------------------------- semantic cache


def test_semantic_cache_hit_and_miss():
    sc = SemanticCache(enabled=True, threshold=0.9)
    emb = np.array([1.0, 0.0, 0.0])
    sc.put("как дела", emb, "M:ru", {"answer": "норм"})
    assert sc.lookup("как дела", emb, "M:ru") == {"answer": "норм"}
    assert sc.lookup("как дела", np.array([0.0, 1.0, 0.0]), "M:ru") is None


def test_semantic_cache_accepts_both_embedding_shapes():
    """Regression: the classifier returns (D,), the batch service (1, D).

    A shape mismatch used to surface in np.dot inside the search and turn into
    a 500 on a live request — on the very second dialogue turn.
    """
    sc = SemanticCache(enabled=True, threshold=0.9)
    flat = np.array([1.0, 0.0, 0.0])
    batched = np.array([[1.0, 0.0, 0.0]])

    sc.put("как дела", batched, "M:ru", {"answer": "норм"})
    assert sc.lookup("как дела", flat, "M:ru") == {"answer": "норм"}

    sc2 = SemanticCache(enabled=True, threshold=0.9)
    sc2.put("как дела", flat, "M:ru", {"answer": "норм"})
    assert sc2.lookup("как дела", batched, "M:ru") == {"answer": "норм"}


def test_semantic_cache_normalizes_legacy_rows(tmp_path):
    """Rows written before normalization are stored in the DB as [[...]].

    Without normalizing on read, a server restart did not fix the crash: an old
    row surfaced on the very first similar request.
    """
    import json
    import sqlite3

    db = tmp_path / "c.db"
    SemanticCache(enabled=True, threshold=0.9, db_path=str(db))
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO semantic_cache VALUES (?,?,?,?,?,0)",
            (
                "M:ru",
                "как дела",
                json.dumps([[1.0, 0.0, 0.0]]),
                json.dumps({"answer": "норм"}),
                time.time(),
            ),
        )

    sc = SemanticCache(enabled=True, threshold=0.9, db_path=str(db))
    hit = sc.lookup("как дела", np.array([1.0, 0.0, 0.0]), "M:ru")
    assert hit == {"answer": "норм"}


def test_semantic_cache_namespace_isolates():
    sc = SemanticCache(enabled=True, threshold=0.9)
    emb = np.array([1.0, 0.0, 0.0])
    sc.put("вопрос", emb, "M:ru", {"answer": "1"})
    assert sc.lookup("вопрос", emb, "L:ru") is None


def test_semantic_cache_skips_recency():
    sc = SemanticCache(enabled=True, threshold=0.9)
    emb = np.array([1.0, 0.0, 0.0])
    sc.put("как дела", emb, "M:ru", {"answer": "норм"})
    assert sc.lookup("как дела сейчас", emb, "M:ru") is None


def test_exact_guard_blocks_swapped_entities():
    """Порядок сущностей значим: «сравни A и B» и «сравни B и A» дают
    почти одинаковый вектор, но разные ответы."""
    a = _exact_sequence("сравни Москву и Париж")
    b = _exact_sequence("сравни Париж и Москву")
    assert not _exact_guard(a, b)  # swapped — block
    assert _exact_guard(a, _exact_sequence("сравни Москву и Париж"))
    c = _exact_sequence("сравни Москву и Лондон")
    assert not _exact_guard(a, c)  # different name — block


def test_exact_tokens_still_compare_as_sets():
    assert _exact_tokens("A1 и B2") == _exact_tokens("B2 и A1")


def test_semantic_cache_disabled_returns_none():
    sc = SemanticCache(enabled=False)
    assert sc.lookup("x", np.array([1.0]), "M:ru") is None
