"""The five-step loop, driven by a scripted model so it runs without GPU."""

import json

from tool_use import executor, tools
from tool_use.executor import ToolLoop, build_system_prompt, run_tool_loop
from tool_use.tools import default_registry


class ScriptedLoop(ToolLoop):
    """A ToolLoop whose 'model' replays a fixed list of generations."""

    def __init__(self, outputs, **kwargs):
        super().__init__(**kwargs)
        self.outputs = list(outputs)
        self.seen: list[list[dict]] = []

    def generate(self, messages):
        self.seen.append([dict(m) for m in messages])
        return self.outputs.pop(0)


def call(name, **arguments):
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


LOAN = {"principal": 10000, "annual_rate_pct": 12, "months": 12}


class EcbReply:
    """A fake ECB response with the Euribor at 3.247 for 2026-09."""

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "dataSets": [{"series": {"0:0": {"observations": {"0": [3.247]}}}}],
            "structure": {"dimensions": {"observation": [{"values": [{"id": "2026-09"}]}]}},
        }


def test_answer_without_tools_ends_in_one_turn():
    loop = ScriptedLoop(["La TAE incluye comisiones; el nominal no."])
    response = loop.run("¿Qué es la TAE?")
    assert response.answer.startswith("La TAE") and response.tool_calls == []
    assert response.turns == 1 and response.model == "Qwen/Qwen3-1.7B"


def test_single_call_result_is_reinjected_with_role_tool():
    loop = ScriptedLoop([call("loan_metrics", **LOAN), "La cuota es 888.49 euros."])
    response = loop.run("cuota de 10000 a 12 meses al 12%")
    record = response.tool_calls[0]
    assert record.ok and record.result["monthly_payment"] == 888.49 and record.latency_ms >= 0
    assert response.answer == "La cuota es 888.49 euros." and response.turns == 2
    second_prompt = loop.seen[1]
    assert [m["role"] for m in second_prompt] == ["system", "user", "assistant", "tool"]
    assert "888.49" in second_prompt[-1]["content"]
    assert second_prompt[-1]["name"] == "loan_metrics"
    assert loop.last_messages[1]["content"] == "cuota de 10000 a 12 meses al 12%"


def test_system_prompt_carries_the_definitions():
    prompt = build_system_prompt(default_registry())
    for name in ("get_euribor", "loan_metrics", "save_credit_report"):
        assert name in prompt
    assert "Additional rules" in prompt
    assert "Additional rules" not in build_system_prompt(default_registry(), guidelines=False)


def test_multi_step_chain_euribor_then_loan(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda url, **kw: EcbReply())
    loop = ScriptedLoop(
        [
            call("get_euribor", tenor="12M"),
            call("loan_metrics", principal=10000, annual_rate_pct=5.247, months=36),
            "La cuota es de unos 301 euros.",
        ]
    )
    response = loop.run("Euribor a 12 meses más 2 puntos, 10000 euros a 36 meses")
    assert [r.name for r in response.tool_calls] == ["get_euribor", "loan_metrics"]
    assert all(r.ok for r in response.tool_calls) and response.turns == 3
    assert "3.247" in loop.seen[1][-1]["content"]  # the euribor result reached the model


def test_parallel_calls_in_one_generation_share_a_turn():
    both = call("loan_metrics", **LOAN) + "\n" + call("loan_metrics", **{**LOAN, "months": 24})
    loop = ScriptedLoop([both, "Listo."])
    response = loop.run("compara 12 y 24 meses")
    assert len(response.tool_calls) == 2 and response.turns == 2
    assert [m["role"] for m in loop.seen[1]][-2:] == ["tool", "tool"]
    first, second = (r.result["monthly_payment"] for r in response.tool_calls)
    assert second < first


def test_malformed_json_becomes_an_observation_and_the_model_recovers():
    loop = ScriptedLoop(
        [
            '<tool_call>{"name": "loan_metrics", "arguments": {oops}}</tool_call>',
            call("loan_metrics", **LOAN),
            "888.49",
        ]
    )
    response = loop.run("cuota")
    assert [r.ok for r in response.tool_calls] == [False, True]
    assert response.tool_calls[0].name == "<invalid>"
    assert "not valid JSON" in loop.seen[1][-1]["content"]


def test_parse_errors_carry_a_hint_about_literal_values_and_dependencies():
    arithmetic = (
        '<tool_call>{"name": "loan_metrics", "arguments": '
        '{"principal": 8000, "annual_rate_pct": 3.247 + 1.5, "months": 24}}</tool_call>'
    )
    loop = ScriptedLoop([arithmetic, "ok"])
    response = loop.run("tae")
    error = response.tool_calls[0].error
    assert "not valid JSON" in error and "write 5.247, not 3.247 + 2" in error
    assert "have not received yet" in loop.seen[1][-1]["content"]


def test_invalid_arguments_are_fed_back_and_corrected():
    loop = ScriptedLoop(
        [
            call("loan_metrics", principal="doce mil", annual_rate_pct=12, months=12),
            call("loan_metrics", **LOAN),
            "888.49",
        ]
    )
    response = loop.run("cuota")
    assert not response.tool_calls[0].ok
    assert "invalid arguments" in response.tool_calls[0].error
    assert "ERROR: invalid arguments" in loop.seen[1][-1]["content"]
    assert response.tool_calls[1].ok


def test_unknown_tool_is_reported_with_the_available_ones():
    loop = ScriptedLoop([call("get_weather", location="Madrid"), "No tengo esa herramienta."])
    response = loop.run("¿qué tiempo hace?")
    assert not response.tool_calls[0].ok
    assert "unknown tool" in response.tool_calls[0].error
    assert "loan_metrics" in loop.seen[1][-1]["content"]


def test_action_is_blocked_when_confirmation_is_denied(tmp_path, monkeypatch):
    monkeypatch.setenv(tools.REPORTS_DIR_ENV, str(tmp_path))
    report = call(
        "save_credit_report",
        applicant_id="CL-1",
        decision="review",
        summary="Ratio 43%",
    )
    loop = ScriptedLoop([report, "Necesito tu confirmación."], confirm=lambda n, a: False)
    response = loop.run("guarda el informe")
    assert not response.tool_calls[0].ok and "NOT executed" in response.tool_calls[0].error
    assert list(tmp_path.iterdir()) == []


def test_action_runs_when_confirmation_is_granted(tmp_path, monkeypatch):
    monkeypatch.setenv(tools.REPORTS_DIR_ENV, str(tmp_path))
    asked = []
    report = call(
        "save_credit_report",
        applicant_id="CL-1",
        decision="review",
        summary="Ratio 43%",
    )
    loop = ScriptedLoop([report, "Guardado."], confirm=lambda n, a: asked.append((n, a)) or True)
    response = loop.run("guarda el informe")
    assert response.tool_calls[0].ok and len(list(tmp_path.glob("CL-1_*.md"))) == 1
    assert asked and asked[0][0] == "save_credit_report"


def test_confirmation_policy_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ARCA_TOOLS_CONFIRM", "deny")
    assert executor.confirm_from_env()("save_credit_report", {}) is False
    monkeypatch.setenv("ARCA_TOOLS_CONFIRM", "auto")
    assert executor.confirm_from_env()("save_credit_report", {}) is True


def test_loop_stops_at_max_turns_and_says_so():
    calls = [call("loan_metrics", **{**LOAN, "months": m}) for m in (12, 24, 36)]
    response = ScriptedLoop(calls).run("cuota", max_turns=3)
    assert response.turns == 3 and len(response.tool_calls) == 3
    assert "máximo de turnos" in response.answer


def test_plain_text_next_to_a_call_is_kept_as_fallback_answer():
    first = "Voy a calcularlo.\n" + call("loan_metrics", **LOAN)
    second = "Voy a calcularlo.\n" + call("loan_metrics", **{**LOAN, "months": 24})
    response = ScriptedLoop([first, second]).run("cuota", max_turns=2)
    assert response.answer == "Voy a calcularlo."


def test_thinking_block_is_ignored_for_the_answer():
    loop = ScriptedLoop(["<think>no hace falta herramienta</think>\nEs un tipo de interés."])
    assert loop.run("¿Qué es el Euribor?").answer == "Es un tipo de interés."


def test_run_tool_loop_uses_the_shared_instance(monkeypatch):
    monkeypatch.setattr(executor, "_loop", ScriptedLoop(["hola"]))
    assert run_tool_loop("hola").answer == "hola"


def test_flags_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ARCA_TOOLS_THINKING", "1")
    monkeypatch.setenv("ARCA_TOOLS_GUIDELINES", "0")
    monkeypatch.setenv("ARCA_TOOLS_FEWSHOT", "0")
    monkeypatch.setenv("ARCA_TOOLS_MODEL", "Qwen/Qwen3-4B")
    loop = ToolLoop()
    assert loop.enable_thinking is True and loop.guidelines is False and loop.fewshot is False
    assert loop.model_id == "Qwen/Qwen3-4B"


def test_dependent_calls_emitted_together_are_caught_and_retried(monkeypatch):
    """The 1.7B failure seen in practice: Euribor and the loan call in one generation."""
    monkeypatch.setattr(tools.requests, "get", lambda url, **kw: EcbReply())
    together = (
        call("get_euribor", tenor="12M")
        + "\n"
        + call("loan_metrics", principal=10000, annual_rate_pct=100.2, months=36)
    )
    loop = ScriptedLoop(
        [
            together,
            call("loan_metrics", principal=10000, annual_rate_pct=5.247, months=36),
            "ok",
        ]
    )
    response = loop.run("Euribor a 12 meses más 2 puntos, 10000 euros a 36 meses")
    assert [r.ok for r in response.tool_calls] == [True, False, True]
    assert "implausible" in loop.seen[1][-1]["content"]
    assert response.tool_calls[-1].result["monthly_payment"] > 0


def test_repeating_a_successful_call_is_not_executed_again(monkeypatch):
    """v2 failure: the model re-called get_euribor three times. Network must be hit once."""
    hits = []

    def fake_get(url, **kwargs):
        hits.append(url)
        return EcbReply()

    monkeypatch.setattr(tools.requests, "get", fake_get)
    euribor = call("get_euribor", tenor="6M")
    loop = ScriptedLoop([euribor, euribor, "Euribor 3.247"])
    response = loop.run("Euribor a 6 meses")
    assert len(hits) == 1
    assert [r.ok for r in response.tool_calls] == [True, False]
    assert "duplicate call" in response.tool_calls[1].error
    assert "3.247" in loop.seen[2][-1]["content"]  # the model is reminded of the result


def test_a_failed_call_may_be_retried_with_the_same_arguments(monkeypatch):
    """Only successful calls count as duplicates: a transient error deserves a retry."""
    answers = iter([ConnectionError("down"), EcbReply()])

    def flaky(url, **kwargs):
        result = next(answers)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(tools.requests, "get", flaky)
    euribor = call("get_euribor", tenor="12M")
    response = ScriptedLoop([euribor, euribor, "ok"]).run("Euribor")
    assert [r.ok for r in response.tool_calls] == [False, True]


def test_guidelines_forbid_dependent_calls_in_one_message():
    assert "Never emit a call whose arguments depend" in build_system_prompt(default_registry())


def test_fewshot_example_is_optional_and_not_taken_from_the_benchmark():
    registry = default_registry()
    with_example = build_system_prompt(registry, guidelines=True, fewshot=True)
    without = build_system_prompt(registry, guidelines=True, fewshot=False)
    assert "Example of a dependent chain" in with_example and "dependent chain" not in without
    assert "Example of a dependent chain" not in build_system_prompt(
        registry, guidelines=False, fewshot=True
    )
    bench_numbers = ("10000 euros a 36", "8000 euros a 24", "Euribor a 6 meses más un")
    assert not any(n in executor.FEWSHOT_EXAMPLE for n in bench_numbers)


def test_tenor_description_maps_spanish_words_to_the_codes():
    description = " ".join(tools.get_euribor.description.split())  # undo the docstring wrapping
    assert "'3M' = three months (trimestral)" in description
    assert "'12M' = one year (anual)" in description