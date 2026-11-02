"""The code-agent comparison, with a scripted brain that writes Python (needs smolagents)."""

import pytest

pytest.importorskip("smolagents")

from agent.compare_codeagent import run_code_agent, summarise, to_smolagents_tool  # noqa: E402
from agent.react_agent import ReActAgent, final_answer, freeze_euribor  # noqa: E402

CODE_REPLY = """Thought: chain both tools in one snippet.
<code>
euribor = get_euribor(tenor="12M")
rate = euribor["rate_pct"] + 3
result = loan_metrics(principal=12000, annual_rate_pct=rate, months=36)
final_answer(result["monthly_payment"])
</code>"""


class Scripted(ReActAgent):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = list(outputs)
        self.registry.register(final_answer)
        freeze_euribor(self.registry)

    def generate(self, messages, brain):
        return self.outputs.pop(0)


def test_registry_tools_become_smolagents_tools_with_the_same_interface():
    agent = Scripted([])
    wrapped = to_smolagents_tool(agent.registry.tools["loan_metrics"], agent.registry)
    assert wrapped.name == "loan_metrics"
    assert set(wrapped.inputs) >= {"principal", "annual_rate_pct", "months", "upfront_fee"}
    assert wrapped.inputs["upfront_fee"]["nullable"] and "nullable" not in wrapped.inputs["months"]
    assert wrapped(principal=1000, annual_rate_pct=0, months=10)["monthly_payment"] == 100.0


def test_a_failing_tool_raises_so_the_code_agent_sees_the_error():
    agent = Scripted([])
    wrapped = to_smolagents_tool(agent.registry.tools["loan_metrics"], agent.registry)
    with pytest.raises(RuntimeError, match="implausible"):
        wrapped(principal=1000, annual_rate_pct=99, months=12)


def test_code_agent_chains_two_tools_in_one_step():
    task = {
        "task": "cuota de 12000 a 36 meses al Euribor 12M + 3",
        "required_tools": ["get_euribor", "loan_metrics"],
        "check": {"type": "number", "value": 362.35, "tolerance": 0.02},
    }
    row = run_code_agent(Scripted([CODE_REPLY]), "base", task, max_steps=4)
    assert row["correct"] and row["required_tools_used"]
    assert row["n_steps"] == 1 and row["tokens_used"] > 0 and not row["error"]


def test_summarise_averages_the_rows():
    rows = [
        {"correct": True, "required_tools_used": True, "n_steps": 2, "tokens_used": 100},
        {"correct": False, "required_tools_used": True, "n_steps": 4, "tokens_used": 300},
    ]
    summary = summarise(rows)
    assert summary["success_rate"] == 0.5 and summary["mean_steps"] == 3
    assert summary["mean_tokens"] == 200 and summary["n_tasks"] == 2
