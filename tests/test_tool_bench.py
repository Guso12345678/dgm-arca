"""The benchmark must be consistent with the registry, or the metrics measure typos."""
 
import json
from collections import Counter
from pathlib import Path
 
from tool_use.evaluate import load_bench, score_case
from tool_use.tools import default_registry
 
BENCH = Path("tool_use/bench/cases.jsonl")
 
 
def test_bench_size_and_negatives():
    cases = load_bench(BENCH)
    assert len(cases) >= 30
    assert sum(not c["needs_tool"] for c in cases) >= 5
    assert len({c["query"] for c in cases}) == len(cases)
 
 
def test_every_tool_is_covered():
    counts = Counter(c["expected_tool"] for c in load_bench(BENCH) if c["needs_tool"])
    assert set(counts) == set(default_registry().tools)
    assert all(n >= 3 for n in counts.values())
 
 
def test_expected_arguments_are_valid_for_the_expected_tool():
    registry = default_registry()
    for case in load_bench(BENCH):
        if not case["needs_tool"]:
            assert "expected_tool" not in case
            continue
        tool = registry.tools[case["expected_tool"]]
        required = set(tool.definition["function"]["parameters"].get("required", []))
        args = case["expected_arguments"]
        assert required <= set(args), case["query"]
        tool.validate(args)  # raises if a type or a literal is wrong
 
 
def test_expected_calls_actually_run(tmp_path, monkeypatch):
    """Every loan/report case must execute with its expected arguments (Euribor needs network)."""
    monkeypatch.setenv("ARCA_REPORTS_DIR", str(tmp_path))
    registry = default_registry()
    for case in load_bench(BENCH):
        if case["needs_tool"] and case["expected_tool"] != "get_euribor":
            record = registry.call(case["expected_tool"], case["expected_arguments"])
            assert record.ok, (case["query"], record.error)
 
 
def test_chains_name_registered_tools_and_start_with_the_expected_tool():
    for case in load_bench(BENCH):
        chain = case.get("expected_chain")
        if chain:
            assert chain[0] == case["expected_tool"]
            assert set(chain) <= set(default_registry().tools)
 
 
def test_scoring_accepts_the_expected_call_and_rejects_a_tool_for_a_negative():
    for case in load_bench(BENCH):
        if case["needs_tool"]:
            call = [{"name": case["expected_tool"], "arguments": case["expected_arguments"]}]
            assert score_case(case, call) == {"selection_ok": True, "arguments_ok": True}
        else:
            assert score_case(case, [])["selection_ok"]
            assert not score_case(case, [{"name": "loan_metrics", "arguments": {}}])["selection_ok"]
 
 
def test_bench_file_is_valid_jsonl():
    for line in BENCH.read_text(encoding="utf-8").splitlines():
        assert isinstance(json.loads(line), dict)
 