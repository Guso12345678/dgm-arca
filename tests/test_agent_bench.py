"""The task benchmark is data, and data rots: these tests keep it honest."""

import json
from pathlib import Path

import pytest

from agent.evaluate import check_answer
from agent.react_agent import ReActAgent, final_answer, freeze_euribor, search_knowledge_base
from tool_use.tools import loan_metrics

BENCH = Path(__file__).parent.parent / "agent" / "bench" / "tasks.jsonl"
TASKS = [json.loads(line) for line in BENCH.read_text(encoding="utf-8").splitlines() if line]
CHECK_KINDS = {"contains", "regex", "number"}


def test_there_are_at_least_twenty_tasks_chaining_two_tools():
    chained = [t for t in TASKS if len(t["required_tools"]) >= 2]
    assert len(chained) >= 20
    assert any(t["kind"] == "impossible" for t in TASKS)


def test_required_tools_exist_in_the_agent_registry():
    agent = ReActAgent()
    agent.registry.register(final_answer, search_knowledge_base)
    for task in TASKS:
        assert set(task["required_tools"]) <= set(agent.registry.tools), task["task"]


def test_every_task_has_a_valid_automatic_check():
    for task in TASKS:
        assert task["check"]["type"] in CHECK_KINDS
        assert task["task"].strip() and task["kind"]
    assert len({t["task"] for t in TASKS}) == len(TASKS)


@pytest.mark.parametrize("task", [t for t in TASKS if "calc" in t and "metric" in t["calc"]])
def test_numeric_answers_match_the_tool_the_agent_must_use(task):
    calc = task["calc"]
    result = loan_metrics.fn(
        principal=calc["principal"],
        annual_rate_pct=calc["rate"],
        months=calc["months"],
        upfront_fee=calc.get("fee", 0.0),
        monthly_income=calc.get("income"),
        other_monthly_debts=calc.get("debts", 0.0),
    )
    assert task["check"]["value"] == pytest.approx(result[calc["metric"]], abs=0.006)


def test_euribor_dependent_tasks_use_the_frozen_table():
    agent = ReActAgent()
    freeze_euribor(agent.registry)
    assert agent.registry.call("get_euribor", {"tenor": "12M"}).result["rate_pct"] == 2.5
    task = next(t for t in TASKS if t["kind"] == "euribor->metrics")
    assert task["calc"]["rate"] == pytest.approx(5.5)  # 2.5 (12M) + 3 points


def test_the_checks_accept_the_right_answer_and_reject_a_wrong_one():
    for task in TASKS:
        check = task["check"]
        if check["type"] == "number":
            assert check_answer(task, f"Resultado: {check['value']}")
            assert not check_answer(task, "Resultado: -1.5")
    impossible = next(t for t in TASKS if t["kind"] == "impossible")
    assert check_answer(impossible, "No puedo conocer ese dato.")
    assert not check_answer(impossible, "Será 3.1 %.")
