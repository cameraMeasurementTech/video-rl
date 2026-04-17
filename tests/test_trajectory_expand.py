from data.trajectory_expand import (
    expand_per_assistant_step,
    outcome_only_prefix,
    prefix_for_next_assistant,
)


def test_prefix_next_assistant_strips_trailing_assistant():
    msgs = [
        {"role": "user", "content": "fix bug"},
        {"role": "assistant", "content": "here is patch"},
    ]
    p = prefix_for_next_assistant(msgs)
    assert len(p) == 1
    assert p[0]["role"] == "user"


def test_per_assistant_step():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    rows = expand_per_assistant_step(msgs)
    assert len(rows) == 2
    assert rows[0][0] == [{"role": "user", "content": "a"}]
    assert rows[1][0][-1]["content"] == "c"


def test_outcome_only_before_last_assistant():
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "final"},
    ]
    p = outcome_only_prefix(msgs)
    assert len(p) == 3
    assert p[-1]["content"] == "u2"
