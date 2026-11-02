"""The ReAct loop, with a scripted brain: no GPU, no network, no model."""

import json
import time

import pytest

from agent.react_agent import (
    ReActAgent,
    call_with_timeout,
    final_answer,
    freeze_euribor,
    render_hits,
    search_knowledge_base,
    set_kb_search,
)
from rag.chunking import Chunk
from rag.retriever import Hit
from tool_use.registry import tool
from tool_use.tools import default_registry


def call(name: str, **arguments) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


class Scripted(ReActAgent):
    """An agent whose brain replays a list of outputs and remembers the prompts it saw."""

    def __init__(self, outputs: list[str], **kwargs):
        super().__init__(**kwargs)
        self.outputs = list(outputs)
        self.prompts: list[list[dict]] = []
        self.registry.register(final_answer, search_knowledge_base)
        freeze_euribor(self.registry)

    def generate(self, messages, brain):
        self.prompts.append(messages)
        return self.outputs.pop(0)


def names(response) -> list[str]:
    return [s.action.name if s.action else "-" for s in response.steps]


def test_euribor_then_loan_then_answer():
    agent = Scripted(
        [
            "Thought: I need the Euribor.\n" + call("get_euribor", tenor="12M"),
            "Thought: 2.5 + 3 = 5.5\n"
            + call("loan_metrics", principal=10000, annual_rate_pct=5.5, months=24),
            call("final_answer", answer="La cuota es 440.45 euros"),
        ]
    )
    response = agent.run("Cuota de 10000 euros a 24 meses al Euribor 12M más 3", brain="base")
    assert response.succeeded and response.n_steps == 3
    assert names(response) == ["get_euribor", "loan_metrics", "final_answer"]
    assert "2.5" in response.steps[0].observation  # the frozen table
    assert response.steps[0].thought == "I need the Euribor."
    assert response.final_answer.endswith("440.45 euros")
    assert response.tokens_used > 0 and response.model
    assert agent.steps == response.steps


def test_only_the_first_call_of_a_message_runs():
    agent = Scripted(
        [
            call("get_euribor", tenor="12M")
            + call("loan_metrics", principal=1000, annual_rate_pct=99, months=12),
            call("final_answer", answer="ok"),
        ]
    )
    response = agent.run("x", brain="base")
    assert names(response) == ["get_euribor", "final_answer"]
    assert "Only your first tool call" in response.steps[0].observation


def test_errors_are_observations_the_model_can_recover_from():
    agent = Scripted(
        [
            call("no_such_tool"),
            call("loan_metrics", principal=1000, annual_rate_pct=5),  # months missing
            '<tool_call>{"name": "loan_metrics", "arguments": {oops}}</tool_call>',
            call("loan_metrics", principal=1000, annual_rate_pct=5, months=12),
            call("final_answer", answer="hecho"),
        ]
    )
    response = agent.run("x", brain="base")
    assert response.succeeded and response.n_steps == 5
    observations = [s.observation for s in response.steps]
    assert "unknown tool" in observations[0]
    assert "invalid arguments" in observations[1]
    assert "not valid JSON" in observations[2] and "literal values" in observations[2]
    assert "monthly_payment" in observations[3]


def test_a_tool_that_hangs_times_out_and_the_run_goes_on():
    @tool
    def slow() -> str:
        """Wait too long."""
        time.sleep(2)
        return "late"

    agent = Scripted([call("slow"), call("final_answer", answer="sin datos")], tool_timeout_s=0.05)
    agent.registry.register(slow)
    response = agent.run("x", brain="base")
    assert response.succeeded
    assert "timeout" in response.steps[0].observation


def test_call_with_timeout_propagates_results_and_errors():
    assert call_with_timeout(lambda: 3, 1) == 3
    with pytest.raises(ZeroDivisionError):
        call_with_timeout(lambda: 1 / 0, 1)


def test_repeating_the_same_call_is_stopped():
    same = call("get_euribor", tenor="12M")
    agent = Scripted([same, same, same, same, same])
    response = agent.run("x", max_steps=8, brain="base")
    assert not response.succeeded and response.final_answer == ""
    assert response.n_steps == 3  # executed, reminded, stopped: it did not burn all 8 steps
    assert "duplicate call" in response.steps[1].observation


def test_prose_after_a_tool_is_accepted_and_flagged():
    agent = Scripted([call("get_euribor", tenor="3M"), "El Euribor a 3 meses es 2.1 %."])
    response = agent.run("x", brain="base")
    assert response.succeeded and response.final_answer.startswith("El Euribor")
    assert "no final_answer call" in response.steps[-1].thought
    assert response.steps[-1].action.name == "final_answer"


def test_prose_without_any_tool_gets_one_reminder_first():
    agent = Scripted(["Son unos 300 euros.", call("get_euribor", tenor="3M"), "2.1"])
    response = agent.run("x", brain="base")
    assert "[reminder]" in response.steps[0].observation
    assert response.n_steps == 3 and response.final_answer == "2.1"


def test_an_answer_in_prose_is_accepted_after_the_reminder_for_impossible_tasks():
    agent = Scripted(["No puedo hacerlo.", "No dispongo de esa información."])
    response = agent.run("x", brain="base")
    assert response.succeeded and response.final_answer == "No dispongo de esa información."


def test_a_cut_off_think_block_is_not_an_answer():
    agent = Scripted(["<think>vamos a ver si", call("final_answer", answer="fin")])
    response = agent.run("x", brain="thinking")
    assert "[reminder]" in response.steps[0].observation and response.final_answer == "fin"


def test_running_out_of_steps_fails_cleanly():
    agent = Scripted([call("get_euribor", tenor="1M"), call("get_euribor", tenor="3M")])
    response = agent.run("x", max_steps=2, brain="base")
    assert not response.succeeded and response.final_answer == "" and response.n_steps == 2


def test_actions_with_side_effects_need_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_REPORTS_DIR", str(tmp_path))
    save = call("save_credit_report", applicant_id="A-1", decision="review", summary="cuota 300")
    denied = Scripted([save, call("final_answer", answer="pido confirmación")])
    denied.confirm = lambda name, arguments: False
    response = denied.run("guarda el informe", brain="base")
    assert "confirmation required" in response.steps[0].observation
    assert not list(tmp_path.iterdir())

    allowed = Scripted([save, call("final_answer", answer="guardado")])
    allowed.confirm = lambda name, arguments: True
    allowed.run("guarda el informe", brain="base")
    assert len(list(tmp_path.iterdir())) == 1


def test_memory_pins_the_task_and_summarises_old_turns_when_over_budget():
    task = "TAREA ORIGINAL: calcula la cuota"
    script = [call("get_euribor", tenor=t) for t in ("1M", "3M", "6M", "12M")]
    script += [call("final_answer", answer="fin")]
    agent = Scripted(script, token_budget=150)
    response = agent.run(task, max_steps=8, brain="base")
    assert response.succeeded
    last_prompt = agent.prompts[-1]
    assert "Summary of the conversation so far" in last_prompt[0]["content"]
    assert last_prompt[1]["content"] == task  # the pinned first message is verbatim
    assert "Result of get_euribor" in last_prompt[0]["content"]


def test_unknown_brain_is_rejected():
    with pytest.raises(ValueError):
        Scripted([]).run("x", brain="gpt")


def test_system_prompt_has_the_tools_and_the_rules():
    prompt = Scripted([]).system_prompt(7)
    for name in ("get_euribor", "loan_metrics", "save_credit_report", "final_answer"):
        assert name in prompt
    assert "search_knowledge_base" in prompt and "at most 7 steps" in prompt
    assert "ONE tool call per message" in prompt


def test_search_knowledge_base_renders_ids_sources_and_clips():
    chunk = Chunk(id="c1", doc_id="d", source="ley.pdf", text="catorce " * 300, index=0)
    set_kb_search(lambda query, top_k: [Hit(chunk, 0.9)][:top_k])
    try:
        out = search_knowledge_base(query="desistimiento", top_k=3)
        assert out.startswith("[c1] (ley.pdf)\ncatorce") and out.endswith("[…]")
        assert len(out) < 800
        set_kb_search(lambda query, top_k: [])
        assert "No passages" in search_knowledge_base(query="nada")
        with pytest.raises(ValueError):
            search_knowledge_base(query="x", top_k=50)
    finally:
        set_kb_search(None)
    assert render_hits([Hit(chunk, 0.5)], max_chars=10).count("[…]") == 1


def test_frozen_euribor_is_reproducible_and_keeps_the_schema():
    registry = default_registry()
    schema = registry.tools["get_euribor"].definition
    freeze_euribor(registry)
    record = registry.call("get_euribor", {"tenor": "6M"})
    assert record.ok and record.result["rate_pct"] == 2.2 and record.result["source"] == "fixture"
    assert registry.tools["get_euribor"].definition == schema


def test_llm_summary_mode_asks_the_brain_for_the_summary():
    class WithSummariser(Scripted):
        def generate(self, messages, brain):
            if messages[0]["content"].startswith("Summarise the conversation"):
                return "RESUMEN DEL MODELO"
            return super().generate(messages, brain)

    script = [call("get_euribor", tenor=t) for t in ("1M", "3M", "6M", "12M")]
    script.append(call("final_answer", answer="fin"))
    agent = WithSummariser(script, token_budget=150, summary_mode="llm")
    response = agent.run("TAREA", max_steps=8, brain="base")
    assert response.succeeded
    assert any("RESUMEN DEL MODELO" in p[0]["content"] for p in agent.prompts)


# --- what the 1.7B and 0.6B brains really write (copied from the first real run) -----------

IMITATED_TRANSCRIPT = (
    "<think>\nplan\n</think>\n\nThought: first the rate.\n"
    '<tool_call>\n{"name": "get_euribor", "arguments": {"tenor": "12M"}}\n</tool_call>\n'
    'Observation: {"rate_pct": 1.25}\n\nThought: now the loan.\n'
    '<tool_call>\n{"name": "final_answer", "arguments": {"answer": "324.50"}}\n</tool_call>'
)


def test_an_imitated_observation_and_everything_after_it_is_discarded():
    from agent.react_agent import clean_model_output

    cleaned = clean_model_output(IMITATED_TRANSCRIPT)
    assert "get_euribor" in cleaned
    assert "324.50" not in cleaned and "Observation" not in cleaned
    assert "plan" in cleaned  # the thinking block is left alone


def test_action_json_is_accepted_as_a_tool_call():
    from agent.react_agent import clean_model_output
    from tool_use.parser import parse_tool_calls

    raw = 'Thought: x\n\nAction: \n{"name": "get_euribor", "arguments": {"tenor": "12M"}}\n'
    calls = parse_tool_calls(clean_model_output(raw)).tool_calls
    assert [(c.name, c.arguments, c.error) for c in calls] == [
        ("get_euribor", {"tenor": "12M"}, None)
    ]


def test_action_name_followed_by_arguments_is_accepted():
    from agent.react_agent import clean_model_output
    from tool_use.parser import parse_tool_calls

    raw = 'Action: final_answer\n{"answer": "La cuota es 362.35"}\nThanks'
    calls = parse_tool_calls(clean_model_output(raw)).tool_calls
    assert calls[0].name == "final_answer"
    assert calls[0].arguments == {"answer": "La cuota es 362.35"}


def test_a_sum_inside_the_json_is_added_up():
    from agent.react_agent import clean_model_output
    from tool_use.parser import parse_tool_calls

    raw = (
        '<tool_call>{"name": "loan_metrics", "arguments": {"principal": 12000, '
        '"annual_rate_pct": 2.5 + 3, "months": 36}}</tool_call>'
    )
    (found,) = parse_tool_calls(clean_model_output(raw)).tool_calls
    assert found.error is None
    assert found.arguments["annual_rate_pct"] == 5.5
    assert found.arguments["principal"] == 12000


def test_a_broken_first_call_does_not_let_a_later_final_answer_run():
    broken = '<tool_call>\n{"name": "loan_metrics", "arguments": {"principal": }}\n</tool_call>'
    later = call("final_answer", answer="invented")
    agent = Scripted(
        [broken + "\n" + later, call("get_euribor", tenor="12M"), call("final_answer", answer="ok")]
    )
    response = agent.run("task", max_steps=5)
    assert response.final_answer == "ok"
    assert "invented" not in [s.observation for s in response.steps]
    assert response.steps[0].observation.startswith("ERROR")
