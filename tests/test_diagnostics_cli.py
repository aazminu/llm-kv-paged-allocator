"""Regression tests for `kv_paged` diagnostics and CLI entrypoint."""

from __future__ import annotations

import json
import pytest

from kv_paged.diagnostics import run_diagnostic_suite, validate_runtime_invariants
from kv_paged.__main__ import main


def test_diagnostic_suite_deterministic():
    r1 = run_diagnostic_suite(seed=42)
    r2 = run_diagnostic_suite(seed=42)
    assert r1["repository"] == "llm-kv-paged-allocator"
    assert r1["package"] == "kv_paged"
    assert r1["status"] == "ok"
    assert r1["metrics"] == r2["metrics"]
    assert r1["metrics"]["invariants_ok"] is True


def test_validate_runtime_invariants():
    res = validate_runtime_invariants(seed=7)
    assert res["passed"] is True
    assert res["reproducible"] is True


def test_negative_seed_raises_value_error():
    with pytest.raises(ValueError, match="non-negative"):
        run_diagnostic_suite(seed=-1)


def test_cli_json_output(capsys):
    rc = main(["--seed", "42", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["repository"] == "llm-kv-paged-allocator"
    assert payload["status"] == "ok"


def test_cli_self_test_mode(capsys):
    rc = main(["--self-test", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["passed"] is True
