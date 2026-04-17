"""
Expand multi-turn ``messages`` into VERL training rows (one prompt per row).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Tuple

from .schema import ChatMessage, ExpansionStrategy

_ASSISTANT_ROLES = frozenset({"assistant"})


def _normalize_messages(messages: Any) -> List[ChatMessage]:
    if messages is None:
        return []
    if isinstance(messages, str):
        raise TypeError("messages must be a list of dicts, not a string")
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages)}")
    out: List[ChatMessage] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise TypeError(f"message[{i}] must be dict, got {type(m)}")
        role = m.get("role")
        if not role:
            raise ValueError(f"message[{i}] missing role")
        out.append({"role": str(role), "content": m.get("content", "")})
    return out


def strip_trailing_assistant(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Remove assistant (and optional trailing tool) messages from the end."""
    m = deepcopy(messages)
    while m and m[-1]["role"] in _ASSISTANT_ROLES | {"tool"}:
        m.pop()
    return m


def prefix_for_next_assistant(messages: List[ChatMessage]) -> List[ChatMessage]:
    """
    Strategy A: prefix ending at last user/tool turn so the policy predicts the next assistant message.
    """
    m = _normalize_messages(messages)
    m = strip_trailing_assistant(m)
    if not m:
        return []
    return m


def expand_per_assistant_step(
    messages: List[ChatMessage],
) -> List[Tuple[List[ChatMessage], int]]:
    """
    Strategy B: for each assistant message at index i, one row with prefix messages[:i].

    Returns list of (prefix_messages, step_index) where step_index is the index of the
    assistant message in the original full trajectory (for logging).
    """
    full = _normalize_messages(messages)
    rows: List[Tuple[List[ChatMessage], int]] = []
    for i, msg in enumerate(full):
        if msg["role"] not in _ASSISTANT_ROLES:
            continue
        prefix = deepcopy(full[:i])
        rows.append((prefix, i))
    return rows


def outcome_only_prefix(messages: List[ChatMessage]) -> List[ChatMessage]:
    """
    Strategy C: prefix immediately before the *last* assistant message (final answer step).
    If there is no assistant message, same as Strategy A prefix.
    """
    full = _normalize_messages(messages)
    last_assistant = None
    for i in range(len(full) - 1, -1, -1):
        if full[i]["role"] in _ASSISTANT_ROLES:
            last_assistant = i
            break
    if last_assistant is None:
        return prefix_for_next_assistant(full)
    return deepcopy(full[:last_assistant])


def expand_trajectory(
    messages: List[ChatMessage],
    strategy: ExpansionStrategy,
) -> List[Tuple[List[ChatMessage], int]]:
    """
    Returns list of (prompt_messages, step_index). step_index is 0 for single-row strategies.
    """
    if strategy == "prefix_next_assistant":
        p = prefix_for_next_assistant(messages)
        if not p:
            return []
        return [(p, 0)]
    if strategy == "per_assistant_step":
        return expand_per_assistant_step(messages)
    if strategy == "outcome_only":
        p = outcome_only_prefix(messages)
        if not p:
            return []
        return [(p, 0)]
    raise ValueError(f"unknown strategy: {strategy}")
