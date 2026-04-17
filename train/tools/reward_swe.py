"""
VERL custom reward for SWE: ``compute_score`` compatible with NaiveRewardManager.

Set ``custom_reward_function.path`` to this file and ``name`` to ``compute_score``.

Environment variables:

  SWE_REWARD_MODE   dry_run | heuristic | harness | subprocess  (default: dry_run)

  **subprocess** (recommended for real SWE signal):

  SWE_REWARD_SCRIPT   Path to an executable. Called as::

      SWE_REWARD_SCRIPT <instance_id> <path_to_prediction.txt>

  The model response is written to the temp file. The script must print one line:
  a float in [0, 1] (or ``1`` / ``0``) to stdout. Non-zero exit → 0 reward.

  **harness** (optional SWE-Bench Python API):

  If ``swebench`` is installed and ``SWE_USE_SWEBENCH=1``, tries ``swebench.harness``
  (best-effort; requires full benchmark wiring — see SWE-Bench docs).

  SWE_HARNESS_CMD     Reserved for future shell-based harness templates.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, Optional


def _normalize_patch(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def heuristic_patch_score(solution_str: str, ground_truth: Optional[str], extra_info: Optional[Dict[str, Any]]) -> float:
    """
    Weak reward without Docker: compare normalized patch-like blocks or length ratio.
    Prefer subprocess or swebench for real SWE training.
    """
    if not solution_str.strip():
        return 0.0
    if extra_info:
        golden = extra_info.get("golden_patch") or extra_info.get("patch")
        if isinstance(golden, str) and golden.strip():
            gn = _normalize_patch(golden)
            sn = _normalize_patch(solution_str)
            if gn and gn in sn:
                return 1.0
            if len(gn) > 20 and len(set(gn.split()) & set(sn.split())) / max(len(set(gn.split())), 1) > 0.4:
                return 0.5
    if ground_truth and str(ground_truth).startswith("patch:"):
        if re.search(r"^diff --git|^--- a/|^\+\+\+ b/", solution_str, re.MULTILINE):
            return 0.3
    return 0.1 if len(solution_str) > 50 else 0.0


def _score_via_subprocess(instance_id: str, solution_str: str) -> Optional[float]:
    script = os.environ.get("SWE_REWARD_SCRIPT", "").strip()
    if not script or not os.path.isfile(script):
        return None
    if not os.access(script, os.X_OK):
        # still try running with interpreter if .py
        pass
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(solution_str)
        pred_path = tmp.name
    try:
        cmd = [script, instance_id, pred_path]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("SWE_REWARD_TIMEOUT", "600")),
            check=False,
        )
        if proc.returncode != 0:
            print(f"[reward_swe] SWE_REWARD_SCRIPT failed rc={proc.returncode} stderr={proc.stderr[:500]}")
            return 0.0
        line = (proc.stdout or "").strip().splitlines()
        if not line:
            return 0.0
        return float(line[-1].strip())
    except (ValueError, subprocess.TimeoutExpired, OSError) as e:
        print(f"[reward_swe] subprocess reward error: {e}")
        return 0.0
    finally:
        try:
            os.unlink(pred_path)
        except OSError:
            pass


def _try_swebench_harness(
    solution_str: str,
    extra_info: Dict[str, Any],
) -> Optional[float]:
    if os.environ.get("SWE_USE_SWEBENCH", "").strip() not in {"1", "true", "yes"}:
        return None
    try:
        # Optional dependency; API varies by swebench version — user should wire their fork.
        import swebench  # noqa: F401
    except ImportError:
        return None
    # Placeholder: real integration maps instance_id + patch to docker eval
    print("[reward_swe] SWE_USE_SWEBENCH set but auto-harness not wired; use SWE_REWARD_SCRIPT.")
    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[str],
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Return scalar reward (typically placed on last token by NaiveRewardManager).
    """
    mode = os.environ.get("SWE_REWARD_MODE", "dry_run").strip().lower()
    extra_info = extra_info or {}
    instance_id = str(extra_info.get("instance_id", "") or "")

    if mode == "subprocess":
        sc = _score_via_subprocess(instance_id, solution_str)
        if sc is not None:
            return float(max(0.0, min(1.0, sc)))
        print("[reward_swe] SWE_REWARD_SCRIPT not set or missing; falling back to heuristic.")
        mode = "heuristic"

    if mode == "harness":
        r = _try_swebench_harness(solution_str, extra_info)
        if r is not None:
            return float(r)
        raise NotImplementedError(
            "SWE_REWARD_MODE=harness requires SWE_USE_SWEBENCH=1 with a wired swebench install, "
            "or use SWE_REWARD_MODE=subprocess with SWE_REWARD_SCRIPT."
        )

    if mode == "dry_run":
        h = hashlib.sha256(solution_str.encode("utf-8", errors="ignore")).hexdigest()
        return (int(h[:8], 16) % 10000) / 10000.0 * 0.2

    if mode == "heuristic":
        return float(heuristic_patch_score(solution_str, ground_truth, extra_info))

    raise ValueError(f"Unknown SWE_REWARD_MODE={mode!r}")


compute_swe_score = compute_score
