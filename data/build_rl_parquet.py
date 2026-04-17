#!/usr/bin/env python3
"""
Build VERL-compatible RL parquet from raw ``instance_id`` + ``messages`` trajectories.

Example:
  python -m data.build_rl_parquet \\
    --input data/raw/trajectories.jsonl \\
    --manifest data/manifests/swe_instances.jsonl \\
    --out-dir data/parquet \\
    --strategy prefix_next_assistant \\
    --val-ratio 0.02
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import datasets  # HuggingFace datasets for parquet export

from .enrich_instance import InstanceManifest, enrich_row, load_manifest
from .schema import DEFAULT_DATA_SOURCE, VERL_PROMPT_KEY, VerlSweRow
from .trajectory_expand import expand_trajectory


def _parse_messages(cell: Any) -> List[Dict[str, Any]]:
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        return json.loads(cell)
    raise TypeError(f"messages must be list or JSON string, got {type(cell)}")


def _normalize_value_for_parquet(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_value_for_parquet(x) for x in value]
    if isinstance(value, dict):
        return {k: _normalize_value_for_parquet(v) for k, v in value.items()}
    return str(value)


def build_rows_for_instance(
    instance_id: str,
    messages: Any,
    manifest: Dict[str, InstanceManifest],
    *,
    strategy: ExpansionStrategy,
    data_source: str,
    drop_missing: bool,
) -> List[VerlSweRow]:
    enriched = enrich_row(instance_id, manifest, drop_missing=drop_missing)
    if enriched is None:
        return []
    reward_model, base_extra = enriched

    msgs = _parse_messages(messages)
    expanded = expand_trajectory(msgs, strategy)
    rows: List[VerlSweRow] = []
    for prompt_messages, step_index in expanded:
        if not prompt_messages:
            continue
        extra = dict(base_extra)
        extra["step_index"] = step_index
        row: VerlSweRow = {
            "instance_id": instance_id,
            "messages": _normalize_value_for_parquet(prompt_messages),
            "data_source": data_source,
            "reward_model": dict(reward_model),
            "extra_info": _normalize_value_for_parquet(extra),
        }
        rows.append(row)
    return rows


def load_input_records(path: Path) -> List[Dict[str, Any]]:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".parquet":
        ds = datasets.load_dataset("parquet", data_files=str(path))["train"]
        return [dict(ds[i]) for i in range(len(ds))]
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def split_by_instance(
    rows: List[VerlSweRow],
    val_ratio: float,
    seed: int,
) -> tuple[List[VerlSweRow], List[VerlSweRow]]:
    ids = sorted({str(r["instance_id"]) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_ratio)
    val_ids = set(ids[:n_val])
    train: List[VerlSweRow] = []
    val: List[VerlSweRow] = []
    for r in rows:
        if str(r["instance_id"]) in val_ids:
            val.append(r)
        else:
            train.append(r)
    return train, val


def main() -> None:
    p = argparse.ArgumentParser(description="Build VERL RL parquet for SWE trajectories")
    p.add_argument("--input", type=Path, required=True, help="JSONL or Parquet with instance_id + messages")
    p.add_argument("--manifest", type=Path, required=True, help="JSON/JSONL manifest keyed by instance_id")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--strategy",
        choices=["prefix_next_assistant", "per_assistant_step", "outcome_only"],
        default="prefix_next_assistant",
    )
    p.add_argument("--data-source", default=DEFAULT_DATA_SOURCE)
    p.add_argument("--keep-missing-manifest", action="store_true", help="Keep rows with empty manifest (weak reward)")
    p.add_argument("--val-ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    raw_rows = load_input_records(args.input)
    out_rows: List[VerlSweRow] = []
    drop_missing = not args.keep_missing_manifest

    for raw in raw_rows:
        iid = raw.get("instance_id")
        if not iid:
            continue
        msgs = raw.get("messages")
        if msgs is None:
            continue
        built = build_rows_for_instance(
            str(iid),
            msgs,
            manifest,
            strategy=args.strategy,
            data_source=args.data_source,
            drop_missing=drop_missing,
        )
        out_rows.extend(built)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not out_rows:
        raise SystemExit(
            "No training rows produced. Check manifest join (--keep-missing-manifest?), "
            "input paths, and trajectory expansion."
        )
    if args.val_ratio > 0:
        train, val = split_by_instance(out_rows, args.val_ratio, args.seed)
    else:
        train, val = out_rows, []

    def to_hf(rows: List[VerlSweRow]) -> datasets.Dataset:
        if not rows:
            return datasets.Dataset.from_dict({})
        # Flatten for HF datasets
        cols: Dict[str, List[Any]] = {k: [] for k in rows[0].keys()}
        for r in rows:
            for k in cols:
                cols[k].append(r[k])
        return datasets.Dataset.from_dict(cols)

    train_ds = to_hf(train)
    train_path = args.out_dir / "train.parquet"
    train_ds.to_parquet(str(train_path))
    print(f"Wrote {len(train)} rows to {train_path} (prompt column: {VERL_PROMPT_KEY})")

    if val:
        val_ds = to_hf(val)
        val_path = args.out_dir / "val.parquet"
        val_ds.to_parquet(str(val_path))
        print(f"Wrote {len(val)} rows to {val_path}")


if __name__ == "__main__":
    main()
