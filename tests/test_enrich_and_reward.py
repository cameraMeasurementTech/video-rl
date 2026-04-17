import json
import tempfile
from pathlib import Path

from data.enrich_instance import enrich_row, load_manifest
from train.tools.reward_swe import compute_score


def test_load_manifest_jsonl():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.jsonl"
        p.write_text(
            json.dumps(
                {
                    "instance_id": "x__y-1",
                    "repo": "x/y",
                    "base_commit": "c0",
                    "patch": "diff --git a/f.py",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        m = load_manifest(p)
        assert "x__y-1" in m
        assert m["x__y-1"].repo == "x/y"


def test_enrich_row():
    from data.enrich_instance import InstanceManifest

    manifest = {
        "i1": InstanceManifest(instance_id="i1", repo="r", base_commit="b", patch="p"),
    }
    r = enrich_row("i1", manifest)
    assert r is not None
    rm, ex = r
    assert rm["ground_truth"].startswith("patch:")
    assert ex["instance_id"] == "i1"


def test_compute_score_dry_run_deterministic():
    a = compute_score("swe", "hello", None, {})
    b = compute_score("swe", "hello", None, {})
    assert a == b
    assert 0.0 <= a <= 1.0


def test_compute_score_heuristic(monkeypatch):
    monkeypatch.setenv("SWE_REWARD_MODE", "heuristic")
    s = compute_score(
        "swe",
        "diff --git a/x b/x\n+foo",
        "patch:1",
        {"golden_patch": "diff --git a/x b/x\n+foo"},
    )
    assert s >= 0.99
