"""
Shared loaders for trajectory tables (Parquet shards, JSONL).
"""

from __future__ import annotations

import glob as glob_mod
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import datasets


def parse_messages(cell: Any) -> List[Dict[str, Any]]:
    """Normalize messages from Parquet/HF (list, JSON string, numpy, pyarrow)."""
    if cell is None:
        raise TypeError("messages is None")
    if isinstance(cell, str):
        return json.loads(cell)
    if isinstance(cell, dict):
        return [cell]
    if isinstance(cell, (list, tuple)):
        out: List[Dict[str, Any]] = []
        for m in cell:
            if isinstance(m, dict):
                out.append(m)
            else:
                out.append(dict(m))
        return out
    try:
        import numpy as np

        if isinstance(cell, np.ndarray):
            return parse_messages(cell.tolist())
    except ImportError:
        pass
    if hasattr(cell, "tolist"):
        return parse_messages(cell.tolist())
    raise TypeError(f"messages must be list or JSON string, got {type(cell)}")


def collect_parquet_files(input_path: Path, *, recursive: bool) -> List[Path]:
    raw = str(input_path.expanduser())
    if "*" in raw or "?" in raw:
        matches = sorted(glob_mod.glob(raw, recursive=recursive or "**" in raw))
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
    Load trajectory rows; normalize ``instance_id`` / ``messages`` column names.

    Returns ``(rows, parquet_source_files)``; parquet list empty for JSONL.
    """
    p = Path(input_path).expanduser()

    if p.is_file() and p.suffix.lower() == ".jsonl":
        return load_jsonl_records(p.resolve()), []

    files = collect_parquet_files(p, recursive=recursive)
    ds = load_parquet_rows(files)
    raw: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        row = dict(ds[i])
        if instance_id_key != "instance_id" and instance_id_key in row:
            row["instance_id"] = row.pop(instance_id_key)
        if messages_key != "messages" and messages_key in row:
            row["messages"] = row.pop(messages_key)
        raw.append(row)

    return raw, files
