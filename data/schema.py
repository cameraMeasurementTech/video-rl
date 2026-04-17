"""
VERL RLHF parquet row schema for SWE trajectories.

RLHFDataset expects (see verl/utils/dataset/rl_dataset.py):
- A column named by prompt_key (default ``prompt``); we use ``messages``.
- non_tensor_batch: reward_model, data_source, extra_info
"""

from __future__ import annotations

from typing import Any, List, Literal, NotRequired, TypedDict


class ChatMessage(TypedDict, total=False):
    role: str
    content: Any


class RewardModelRow(TypedDict, total=False):
    """Stored under parquet column ``reward_model`` (nested struct in some writers)."""

    ground_truth: str
    style: str


class VerlSweRow(TypedDict, total=False):
    """One training row after enrichment + expansion."""

    instance_id: str
    messages: List[ChatMessage]
    data_source: str
    reward_model: RewardModelRow
    extra_info: dict[str, Any]


# Column name passed to VERL as data.prompt_key
VERL_PROMPT_KEY = "messages"

DEFAULT_DATA_SOURCE = "swe_bench"

ExpansionStrategy = Literal["prefix_next_assistant", "per_assistant_step", "outcome_only"]
