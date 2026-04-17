"""
Join ``instance_id`` to SWE / SWE-Bench-style **manifest** records for rewards and ``extra_info``.

**Trajectory tables** (what you usually have) only need:

- ``instance_id``
- ``messages`` (multi-turn chat)

They do **not** need ``repo``, ``base_commit``, or ``patch`` columns. Those fields live in a
**separate manifest** JSON/JSONL file (e.g. exported from the SWE-Bench dataset by
``instance_id``), passed to ``build_*_parquet --manifest``. For imitation-only SFT with no
manifest, use ``--skip-manifest`` (see ``minimal_enrichment``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional, Tuple

from .schema import RewardModelRow


@dataclass
class InstanceManifest:
    """
    Minimal fields for ``extra_info`` + optional ground_truth.

    Extend with any keys your ``compute_score`` / Docker harness needs.
    """

    instance_id: str
    repo: str = ""
    base_commit: str = ""
    patch: str = ""
    test_patch: str = ""
    problem_statement: str = ""
    # Opaque blob for your verifier (image, test cmd, env vars, etc.)
    harness: Dict[str, Any] = field(default_factory=dict)

    def to_extra_info(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "harness": dict(self.harness),
        }
        return out

    def ground_truth_placeholder(self) -> str:
        """Default ground_truth string; override if you store a canonical patch hash."""
        if self.patch:
            return f"patch:{len(self.patch)}"
        return "swe:verify"


def _iter_manifest_objects(path: Path) -> Iterator[MutableMapping[str, Any]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl" or path.name.endswith(".jsonl"):
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            if isinstance(obj, dict):
                yield obj
        return
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                yield item
            else:
                raise ValueError(f"Expected list of dicts in {path}, got {type(item)} at index {i}")
    elif isinstance(data, dict):
        yield data
    else:
        raise ValueError(f"Expected dict or list in {path}, got {type(data)}")


def manifest_dict_to_dataclass(row: Mapping[str, Any]) -> InstanceManifest:
    iid = row.get("instance_id")
    if not iid:
        raise ValueError("manifest row missing instance_id")
    harness = row.get("harness")
    if not isinstance(harness, dict):
        harness = {}
    return InstanceManifest(
        instance_id=str(iid),
        repo=str(row.get("repo", "") or ""),
        base_commit=str(row.get("base_commit", "") or ""),
        patch=str(row.get("patch", "") or ""),
        test_patch=str(row.get("test_patch", "") or ""),
        problem_statement=str(row.get("problem_statement", "") or ""),
        harness={k: v for k, v in harness.items()},
    )


def load_manifest(path: str | Path) -> Dict[str, InstanceManifest]:
    """
    Load a JSON/JSONL manifest keyed by ``instance_id``.

    Each line/object may contain standard SWE-Bench fields; unknown keys are
    folded into ``harness`` if you pass a flat JSON (optional helper below).
    """
    path = Path(path)
    out: Dict[str, InstanceManifest] = {}
    for row in _iter_manifest_objects(path):
        m = manifest_dict_to_dataclass(row)
        out[m.instance_id] = m
    return out


def load_manifest_flat_jsonl(path: str | Path) -> Dict[str, InstanceManifest]:
    """
    Same as ``load_manifest`` but merges any keys other than known InstanceManifest
    fields into ``harness`` for convenience.
    """
    known = {
        "instance_id",
        "repo",
        "base_commit",
        "patch",
        "test_patch",
        "problem_statement",
        "harness",
    }
    path = Path(path)
    out: Dict[str, InstanceManifest] = {}
    for row in _iter_manifest_objects(path):
        if "instance_id" not in row:
            continue
        harness = row.get("harness")
        if not isinstance(harness, dict):
            harness = {}
        extra = {k: v for k, v in row.items() if k not in known}
        merged = {**harness, **extra}
        m = manifest_dict_to_dataclass({**row, "harness": merged})
        out[m.instance_id] = m
    return out


def minimal_enrichment(instance_id: str) -> Tuple[RewardModelRow, Dict[str, Any]]:
    """
    Build ``reward_model`` / ``extra_info`` when the trajectory table has **only**
    ``instance_id`` + ``messages`` and you are **not** joining an external manifest.

    Use for LoRA SFT on trajectories alone, or as a stub before you add a manifest /
    ``SWE_REWARD_SCRIPT`` for GRPO.
    """
    rm: RewardModelRow = {"ground_truth": "", "style": "no_manifest"}
    ex: Dict[str, Any] = {
        "instance_id": instance_id,
        "benchmark_metadata": "not_in_trajectory_table",
    }
    return rm, ex


def enrich_row(
    instance_id: str,
    manifest: Mapping[str, InstanceManifest],
    *,
    drop_missing: bool = True,
) -> Optional[Tuple[RewardModelRow, Dict[str, Any]]]:
    """
    Return (reward_model, extra_info) for VERL, or None if missing and drop_missing.
    """
    rec = manifest.get(instance_id)
    if rec is None:
        if drop_missing:
            return None
        rm: RewardModelRow = {"ground_truth": "", "style": "unverified"}
        ex: Dict[str, Any] = {"instance_id": instance_id, "missing_manifest": True}
        return rm, ex
    rm = {
        "ground_truth": rec.ground_truth_placeholder(),
        "style": "rule",
    }
    return rm, rec.to_extra_info()
