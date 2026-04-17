#!/usr/bin/env python3
"""
Build VERL-compatible RL parquet from raw ``instance_id`` + ``messages`` trajectories.

Input can be:

- A **single** ``.parquet`` file
- A **directory** containing one or more ``*.parquet`` shards (e.g. 47 files / 66k rows)
- A **glob** path (e.g. ``data/raw/part-*.parquet``)

JSONL is still supported for small tests.

Example (directory of parquet shards):

  python -m data.build_rl_parquet \\
    --input data/raw/trajectory_shards/ \\
    --manifest data/manifests/swe_instances.jsonl \\
    --out-dir data/parquet \\
    --strategy prefix_next_assistant \\
    --val-ratio 0.02
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, cast

import datasets  # HuggingFace datasets for parquet export

from .enrich_instance import InstanceManifest, enrich_row, load_manifest, minimal_enrichment
from .schema import DEFAULT_DATA_SOURCE, VERL_PROMPT_KEY, ExpansionStrategy, VerlSweRow
from .trajectory_expand import expand_trajectory
from .trajectory_io import load_input_records, parse_messages


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
    skip_manifest: bool = False,
) -> List[VerlSweRow]:
    if skip_manifest:
        reward_model, base_extra = minimal_enrichment(instance_id)
    else:
        enriched = enrich_row(instance_id, manifest, drop_missing=drop_missing)
        if enriched is None:
            return []
        reward_model, base_extra = enriched

    msgs = parse_messages(messages)
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
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Parquet file, directory of .parquet shards, glob (e.g. data/part-*.parquet), or .jsonl",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="When --input is a directory, include **/*.parquet in subdirectories",
    )
    p.add_argument(
        "--instance-id-column",
        default="instance_id",
        metavar="NAME",
        help="Column name for instance id in Parquet (default: instance_id)",
    )
    p.add_argument(
        "--messages-column",
        default="messages",
        metavar="NAME",
        help="Column name for chat messages in Parquet (default: messages)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON/JSONL manifest (repo/commit/patch by instance_id). Omit with --skip-manifest.",
    )
    p.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Trajectory has only instance_id + messages; no join (stub extra_info; use manifest later for GRPO rewards).",
    )
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
    if not args.skip_manifest:
        if args.manifest is None:
            raise SystemExit("Provide --manifest PATH or use --skip-manifest for trajectory-only data.")
        manifest = load_manifest(args.manifest)
    else:
        manifest = {}
    raw_rows, parquet_sources = load_input_records(
        args.input,
        recursive=args.recursive,
        instance_id_key=args.instance_id_column,
        messages_key=args.messages_column,
    )
    if parquet_sources:
        print(f"Loaded {len(raw_rows)} rows from {len(parquet_sources)} parquet file(s).")
    else:
        print(f"Loaded {len(raw_rows)} rows from JSONL.")

    out_rows: List[VerlSweRow] = []
    drop_missing = not args.keep_missing_manifest
    skip_manifest = args.skip_manifest

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
            strategy=cast(ExpansionStrategy, args.strategy),
            data_source=args.data_source,
            drop_missing=drop_missing,
            skip_manifest=skip_manifest,
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
