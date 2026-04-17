#!/usr/bin/env python3
"""
Build VERL **multi-turn SFT** parquet (``messages`` column) for LoRA SFT.

Uses VERL ``MultiTurnSFTDataset`` when ``data.multiturn.enable=true`` in ``fsdp_sft_trainer``:
each row is a full conversation; loss is applied on assistant tokens only.

Example:

  python -m data.build_sft_parquet \\
    --input path/to/47_parquet_shards/ \\
    --manifest path/to/swe_manifest.jsonl \\
    --out-dir data/parquet_sft \\
    --val-ratio 0.02
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

import datasets

from .enrich_instance import enrich_row, load_manifest, minimal_enrichment
from .schema import DEFAULT_DATA_SOURCE
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


def split_rows_by_instance(
    rows: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ids = sorted({str(r["instance_id"]) for r in rows if r.get("instance_id")})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_ratio)
    val_ids = set(ids[:n_val])
    train, val = [], []
    for r in rows:
        iid = str(r.get("instance_id", ""))
        if iid in val_ids:
            val.append(r)
        else:
            train.append(r)
    return train, val


def main() -> None:
    p = argparse.ArgumentParser(description="Build multi-turn SFT parquet for VERL LoRA SFT")
    p.add_argument("--input", type=Path, required=True, help="Parquet dir/file/glob or JSONL")
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest JSON/JSONL (repo/commit/patch by instance_id). Omit with --skip-manifest.",
    )
    p.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Only instance_id + messages in data; no manifest join (SFT on trajectories alone).",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--instance-id-column", default="instance_id")
    p.add_argument("--messages-column", default="messages")
    p.add_argument("--data-source", default=DEFAULT_DATA_SOURCE)
    p.add_argument("--keep-missing-manifest", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.skip_manifest:
        if args.manifest is None:
            raise SystemExit("Provide --manifest PATH or use --skip-manifest.")
        manifest = load_manifest(args.manifest)
    else:
        manifest = {}

    raw_rows, sources = load_input_records(
        args.input,
        recursive=args.recursive,
        instance_id_key=args.instance_id_column,
        messages_key=args.messages_column,
    )
    drop_missing = not args.keep_missing_manifest
    out_rows: List[Dict[str, Any]] = []

    for raw in raw_rows:
        iid = raw.get("instance_id")
        if not iid:
            continue
        if args.skip_manifest:
            _rm, extra = minimal_enrichment(str(iid))
        else:
            enriched = enrich_row(str(iid), manifest, drop_missing=drop_missing)
            if enriched is None:
                continue
            _rm, extra = enriched
        try:
            msgs = parse_messages(raw.get("messages"))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Bad messages for instance_id={iid}: {e}") from e
        if not msgs:
            continue
        # Require at least one assistant turn for SFT signal
        if not any(m.get("role") == "assistant" for m in msgs):
            continue

        row = {
            "instance_id": str(iid),
            "messages": _normalize_value_for_parquet(msgs),
            "data_source": args.data_source,
            "extra_info": _normalize_value_for_parquet({"instance_id": str(iid), **extra}),
        }
        out_rows.append(row)

    if not out_rows:
        raise SystemExit(
            "No SFT rows produced. Check manifest join, messages (need assistant turns), "
            "and --keep-missing-manifest if needed."
        )

    if sources:
        print(f"Source: {len(sources)} parquet file(s), {len(out_rows)} rows after manifest filter.")

    if args.val_ratio > 0:
        train, val = split_rows_by_instance(out_rows, args.val_ratio, args.seed)
    else:
        train, val = out_rows, []

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def to_ds(rows: List[Dict[str, Any]]) -> datasets.Dataset:
        if not rows:
            return datasets.Dataset.from_dict({})
        cols: Dict[str, List[Any]] = {k: [] for k in rows[0]}
        for r in rows:
            for k in cols:
                cols[k].append(r[k])
        return datasets.Dataset.from_dict(cols)

    train_ds = to_ds(train)
    train_path = args.out_dir / "train.parquet"
    train_ds.to_parquet(str(train_path))
    print(f"Wrote {len(train)} rows to {train_path} (column `messages` for multiturn SFT)")

    if val:
        val_ds = to_ds(val)
        val_path = args.out_dir / "val.parquet"
        val_ds.to_parquet(str(val_path))
        print(f"Wrote {len(val)} rows to {val_path}")


if __name__ == "__main__":
    main()
