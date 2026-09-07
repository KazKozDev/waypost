"""Policy A/B and multimodal decomposition.

Every knob in this router was set by an argument. These are the two
pieces that let an argument be replaced by a number, and let a weak
vision pool borrow a strong text one.
"""
import pytest

from waypost.decompose import (
    analyze_multimodal_decomposition,
    decomposition_gain,
    is_deictic_prompt,
    strip_images,
)
from waypost.experiment import Arm, Experiment, ExperimentRegistry, compare
from waypost.schemas import ChatMessage, ChatRequest


def two_arms(**kw):
    return Experiment(
        "t", [Arm("a", {"stochastic_routing": False}), Arm("b", {"stochastic_routing": True})],
        **kw,
    )


# ------------------------------------------------------------ assignment


def test_a_session_stays_in_one_arm():
    """Flipping a conversation mid-way measures neither arm, and destroys
    the prefix stickiness both of them rely on."""
    exp = two_arms()
    first = exp.assign("session-42").name
    assert all(exp.assign("session-42").name == first for _ in range(100))


def test_assignment_survives_a_restart():
    """A hash, not a counter: the same unit lands in the same arm in a
    fresh process, and identically across replicas."""
    assert two_arms().assign("s1").name == two_arms().assign("s1").name


def test_both_arms_get_traffic():
    exp = two_arms()
    names = {exp.assign(f"s{i}").name for i in range(200)}
    assert names == {"a", "b"}


def test_the_split_is_roughly_even():
    exp = two_arms()
    a = sum(1 for i in range(1000) if exp.assign(f"s{i}").name == "a")
    assert 400 < a < 600


def test_weights_shift_the_split():
    exp = Experiment("t", [Arm("small", weight=0.1), Arm("big", weight=0.9)])
    small = sum(1 for i in range(1000) if exp.assign(f"s{i}").name == "small")
    assert 50 < small < 160


def test_an_unassignable_request_goes_to_control():
    """A one-off with no session should not quietly become data for an arm
    it was never in."""
    assert two_arms().assign(None).name == "a"


def test_a_disabled_experiment_assigns_nothing():
    assert two_arms(enabled=False).assign("s1").name == "control"


def test_overrides_reach_the_router():
    reg = ExperimentRegistry([two_arms()])
    got = reg.overrides("s1")
    assert "stochastic_routing" in got
    assert isinstance(got["stochastic_routing"], bool)


def test_colliding_experiments_are_reported(caplog):
    """Two experiments on one knob is a mistake: the second would silently
    win and both reports would be wrong."""
    reg = ExperimentRegistry(
        [
            Experiment("one", [Arm("x", {"stochastic_routing": True})]),
            Experiment("two", [Arm("y", {"stochastic_routing": False})]),
        ]
    )
    with caplog.at_level("ERROR"):
        reg.overrides("s1")
    assert any("collide" in r.message for r in caplog.records)


# ------------------------------------------------------------ comparison


def rows(arm, n, ok, latency, attempts):
    return [
        {
            "arm": arm,
            "outcome": "pass" if i < ok else "fail",
            "latency_ms": latency,
            "attempt_no": attempts,
        }
        for i in range(n)
    ]


def test_comparison_reports_what_each_arm_cost():
    out = compare(rows("a", 10, 8, 300, 2) + rows("b", 10, 9, 100, 1))
    assert out["a"]["success_rate"] == 0.8
    assert out["b"]["p95_latency_ms"] == 100
    assert out["b"]["mean_attempts"] == 1.0
    assert out["_verdict"]["deltas"]["b"]["p95_latency_ms"] == -200


def test_the_sample_size_is_always_reported():
    """Two arms differing by three points over ninety requests differ by
    nothing, and the report has to make that visible."""
    out = compare(rows("a", 3, 3, 100, 1) + rows("b", 2, 1, 100, 1))
    assert out["a"]["n"] == 3 and out["b"]["n"] == 2
    assert "sample sizes" in out["_verdict"]["note"]


def test_unlabelled_rows_count_as_control():
    out = compare([{"outcome": "pass", "latency_ms": 10, "attempt_no": 1}])
    assert out["control"]["n"] == 1


def test_no_verdict_without_a_comparison():
    assert "_verdict" not in compare(rows("a", 5, 5, 10, 1))


# --------------------------------------------------------- decomposition


def img_req(text):
    return ChatRequest(
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                ],
            )
        ]
    )


def test_a_question_about_an_image_is_separable():
    task = analyze_multimodal_decomposition(img_req("Что не так с этой архитектурой?"))
    assert task.is_separable and task.stage == "extraction_first"


def test_a_deictic_question_is_not():
    """"What is circled here" cannot survive losing the image: the
    description would have to already know what was being pointed at."""
    for text in ("what is circled in red?", "объясни что обведено", "this part is odd"):
        assert is_deictic_prompt(text), text
        assert not analyze_multimodal_decomposition(img_req(text)).is_separable


def test_a_text_request_is_never_decomposed():
    r = ChatRequest(messages=[ChatMessage(role="user", content="просто вопрос")])
    assert not analyze_multimodal_decomposition(r).is_separable


def test_stripping_removes_the_image_and_keeps_the_question():
    """The blocks are removed rather than kept alongside: the point is to
    reach a pool that has no vision at all, and leaving them in would
    filter that pool straight back out."""
    out = strip_images(img_req("Что не так?"), "Схема: клиент → одна БД.")
    content = out.messages[0].content
    assert isinstance(content, str)
    assert "Что не так?" in content
    assert "одна БД" in content


def test_splitting_pays_only_when_the_text_pool_is_much_stronger():
    """It costs a second call and loses whatever the description missed.
    Where a vision model can carry the task, end-to-end is cheaper and
    more faithful."""
    assert decomposition_gain(best_vision_quality=0.5, best_text_quality=0.9)
    assert not decomposition_gain(best_vision_quality=0.80, best_text_quality=0.85)
    assert not decomposition_gain(best_vision_quality=0.9, best_text_quality=0.5)


@pytest.mark.parametrize(
    "task_class,expected",
    [("ocr", 1120), ("diagram", 560), ("classification", 70), ("chat", 280)],
)
def test_the_vision_budget_follows_the_task(task_class, expected):
    from waypost.decompose import get_vision_token_budget

    assert get_vision_token_budget(task_class) == expected
