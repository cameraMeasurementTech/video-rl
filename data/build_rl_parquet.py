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
import glob as glob_mod
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import datasets  # HuggingFace datasets for parquet export

from .enrich_instance import InstanceManifest, enrich_row, load_manifest
from .schema import DEFAULT_DATA_SOURCE, VERL_PROMPT_KEY, ExpansionStrategy, VerlSweRow
from .trajectory_expand import expand_trajectory


def _parse_messages(cell: Any) -> List[Dict[str, Any]]:
    """Normalize messages from Parquet/HF (list, JSON string, numpy, pyarrow struct arrays)."""
    if cell is None:
        raise TypeError("messages is None")
    if isinstance(cell, str):
        return json.loads(cell)
    if isinstance(cell, dict):
        # single message object by mistake
        return [cell]
    if isinstance(cell, (list, tuple)):
        out: List[Dict[str, Any]] = []
        for m in cell:
            if isinstance(m, dict):
                out.append(m)
            else:
                out.append(dict(m))  # Mapping-like (e.g. pandas Series)
        return out
    # numpy ndarray
    try:
        import numpy as np

        if isinstance(cell, np.ndarray):
            return _parse_messages(cell.tolist())
    except ImportError:
        pass
    if hasattr(cell, "tolist"):
        return _parse_messages(cell.tolist())
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


def collect_parquet_files(input_path: Path, *, recursive: bool) -> List[Path]:
    """
    Resolve ``input_path`` to a sorted list of ``.parquet`` files.

    - **Glob** (path contains ``*`` or ``?``): expand with :func:`glob.glob`.
    - **Single file**: must end with ``.parquet``.
    - **Directory**: all ``*.parquet`` in that directory; with ``recursive``, use ``**/*.parquet``.
    """
    raw = str(input_path.expanduser())
    if "*" in raw or "?" in raw:
        matches = sorted(
            glob_mod.glob(raw, recursive=recursive or "**" in raw)
        )
        files = [Path(m) for m in matches if str(m).lower().endswith(".parquet")]
        if not files:
            raise FileNotFoundError(f"No .parquet files matched pattern: {input_path}")
        return files

    path = Path(raw).resolve()
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a .parquet file, got: {path}")
        return [path]
    if path.is_dir():
        pattern = "**/*.parquet" if recursive else "*.parquet"
        files = sorted(path.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No .parquet files under {path} (pattern {pattern!r}). "
                "Use --recursive to include subdirectories."
            )
        return files
    raise FileNotFoundError(f"Not a file, directory, or glob pattern: {input_path}")


def load_parquet_rows(parquet_files: Sequence[Path]) -> datasets.Dataset:
    """Load and concatenate multiple Parquet files into one HF Dataset."""
    paths = [str(Path(p).expanduser().resolve()) for p in parquet_files]
    if len(paths) == 1:
        ds = datasets.load_dataset("parquet", data_files=paths[0], split="train")
    else:
        ds = datasets.load_dataset("parquet", data_files=paths, split="train")
    return ds


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    path = path.expanduser().resolve()
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


def load_input_records(
    input_path: Path,
    *,
    recursive: bool = False,
    instance_id_key: str = "instance_id",
    messages_key: str = "messages",
) -> tuple[List[Dict[str, Any]], List[Path]]:
    """
    Load trajectory rows from Parquet shard(s), a single Parquet file, or JSONL.

    Parquet rows are normalized so keys match ``instance_id`` and ``messages`` (aliases remapped).

    Returns ``(rows, parquet_source_files)``. For JSONL input, ``parquet_source_files`` is empty.
    """
    p = Path(input_path).expanduser()

    if p.is_file() and p.suffix.lower() == ".jsonl":
        return load_jsonl_records(p.resolve()), []

    files = collect_parquet_files(p, recursive=recursive)
    ds = load_parquet_rows(files)
    n = len(ds)
    raw: List[Dict[str, Any]] = []
    for i in range(n):
        row = dict(ds[i])
        if instance_id_key != "instance_id" and instance_id_key in row:
            row["instance_id"] = row.pop(instance_id_key)
        if messages_key != "messages" and messages_key in row:
            row["messages"] = row.pop(messages_key)
        raw.append(row)

    return raw, files


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
