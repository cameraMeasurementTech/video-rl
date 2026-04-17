"""
VERL custom reward for SWE: ``compute_score`` compatible with NaiveRewardManager.

Set ``custom_reward_function.path`` to this file and ``name`` to ``compute_score``.

Environment variables (optional):
  SWE_REWARD_MODE   dry_run | heuristic | harness (default: dry_run)
  SWE_HARNESS_CMD   shell command template with {instance_id} {solution_path} — future use
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, Optional


def _normalize_patch(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def heuristic_patch_score(solution_str: str, ground_truth: Optional[str], extra_info: Optional[Dict[str, Any]]) -> float:
    """
    Weak reward without Docker: compare normalized patch-like blocks or length ratio.
    Prefer replacing this with a real SWE-Bench harness call.
    """
    if not solution_str.strip():
        return 0.0
    # If manifest included a golden patch, reward substring similarity (very rough)
    if extra_info:
        golden = extra_info.get("golden_patch") or extra_info.get("patch")
        if isinstance(golden, str) and golden.strip():
            gn = _normalize_patch(golden)
            sn = _normalize_patch(solution_str)
            if gn and gn in sn:
                return 1.0
            # diff-style unified hunk overlap
            if len(gn) > 20 and len(set(gn.split()) & set(sn.split())) / max(len(set(gn.split())), 1) > 0.4:
                return 0.5
    if ground_truth and str(ground_truth).startswith("patch:"):
        # placeholder from InstanceManifest — give partial credit for non-empty diff markers
        if re.search(r"^diff --git|^--- a/|^\+\+\+ b/", solution_str, re.MULTILINE):
            return 0.3
    return 0.1 if len(solution_str) > 50 else 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[str],
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Return scalar reward (typically placed on last token by NaiveRewardManager).

    For production, replace ``dry_run`` / ``heuristic`` with your containerized
    SWE-Bench evaluation using ``extra_info["instance_id"]`` and manifest fields.
    """
    mode = os.environ.get("SWE_REWARD_MODE", "dry_run").strip().lower()
    extra_info = extra_info or {}

    if mode == "dry_run":
        # Deterministic pseudo-score for pipeline testing (not for real training quality)
        h = hashlib.sha256(solution_str.encode("utf-8", errors="ignore")).hexdigest()
        return (int(h[:8], 16) % 10000) / 10000.0 * 0.2

    if mode == "heuristic":
        return float(heuristic_patch_score(solution_str, ground_truth, extra_info))

    if mode == "harness":
        # Hook: call your evaluator (Docker / swebench) — not shipped to avoid heavy deps
        raise NotImplementedError(
            "SWE_REWARD_MODE=harness requires wiring swebench.harness.run_eval or your Docker runner; "
            f"see extra_info keys: {list(extra_info.keys())}"
        )

    raise ValueError(f"Unknown SWE_REWARD_MODE={mode!r}")


# Alias for Hydra configs that use a project-specific name
compute_swe_score = compute_score
