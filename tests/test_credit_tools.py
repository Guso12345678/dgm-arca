"""Phase-2 domain tools: credit maths, Euribor parsing, report saving. No network needed."""

import json
from pathlib import Path

import pytest

from tool_use import credit_math, tools
from tool_use.registry import ToolRegistry
from tool_use.tools import (
    default_registry,
    get_euribor,
    loan_metrics,
    parse_ecb_observation,
    save_credit_report,
)

DATA = Path("rlm/data")


def _rows(name: str) -> list[dict]:
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{path} not found")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- credit maths


def test_payment_known_value():
    # 10000 at 12% nominal over 12 months: textbook 888.49
    assert round(credit_math.monthly_payment(10000, 12, 12), 2) == 888.49


def test_zero_rate_is_straight_division_and_zero_apr():
    payment = credit_math.monthly_payment(1200, 0, 12)
    assert payment == 100
    assert credit_math.apr_pct(1200, 0, payment, 12) == 0.0


def test_apr_without_fee_is_effective_annual_rate():
    payment = round(credit_math.monthly_payment(10000, 12, 12), 2)
    # 12% nominal, monthly compounding -> (1.01)^12 - 1 = 12.68%
    assert credit_math.apr_pct(10000, 0, payment, 12) == pytest.approx(12.68, abs=0.01)


def test_apr_grows_with_the_fee():
    payment = round(credit_math.monthly_payment(10000, 12, 12), 2)
    assert credit_math.apr_pct(10000, 200, payment, 12) > credit_math.apr_pct(10000, 0, payment, 12)


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError):
        credit_math.monthly_payment(-5, 8, 12)
    with pytest.raises(ValueError):
        credit_math.monthly_payment(1000, 8, 0)
    with pytest.raises(ValueError):
        credit_math.debt_ratio_pct(100, 0, 0)
    with pytest.raises(ValueError):
        credit_math.apr_pct(1000, 0, 10, 12)  # repays 120 < 1000: no positive APR


@pytest.mark.parametrize("filename", ["rlm_data_test.jsonl", "rlm_data_train.jsonl"])
def test_tool_reproduces_every_answer_of_the_phase1_dataset(filename):
    """loan_metrics must agree with the verified answers of all four problem types."""
    key = {
        "cuota": "monthly_payment",
        "coste_total": "total_cost",
        "ratio_endeudamiento": "debt_ratio_pct",
        "tae": "apr_pct",
    }
    for row in _rows(filename):
        p = row["_debug_prestamo"]
        metrics = loan_metrics.fn(
            principal=p["principal"],
            annual_rate_pct=p["tipo_nominal"],
            months=p["meses"],
            upfront_fee=p["comision_apertura"],
            monthly_income=p["ingresos_netos"],
            other_monthly_debts=p["otras_deudas"],
        )
        got = f"{metrics[key[row['tipo']]]:.2f}"
        assert got == row["answer"], (row["id"], got, row["answer"])


# ------------------------------------------------------------------ loan_metrics as a tool


def test_loan_metrics_through_the_registry_and_optional_ratio():
    registry = ToolRegistry().register(loan_metrics)
    record = registry.call(
        "loan_metrics", {"principal": 10000, "annual_rate_pct": 12, "months": 12}
    )
    assert record.ok
    assert record.result["monthly_payment"] == 888.49
    assert record.result["debt_ratio_pct"] is None
    with_income = registry.call(
        "loan_metrics",
        {"principal": 10000, "annual_rate_pct": 12, "months": 12, "monthly_income": 2000},
    ).result
    assert with_income["debt_ratio_pct"] == pytest.approx(44.42, abs=0.01)


def test_loan_metrics_uses_a_known_payment_when_given():
    metrics = loan_metrics.fn(principal=5000, annual_rate_pct=9, months=24, monthly_payment=230.0)
    assert metrics["monthly_payment"] == 230.0
    assert metrics["total_cost"] == pytest.approx(230 * 24 - 5000)


def test_loan_metrics_errors_become_observations_for_the_model():
    registry = ToolRegistry().register(loan_metrics)
    wrong_type = registry.call(
        "loan_metrics", {"principal": "twelve thousand", "annual_rate_pct": 8, "months": 12}
    )
    assert not wrong_type.ok and "invalid arguments" in wrong_type.error
    missing = registry.call("loan_metrics", {"principal": 1000})
    assert not missing.ok and "invalid arguments" in missing.error
    bad_value = registry.call(
        "loan_metrics", {"principal": 1000, "annual_rate_pct": 8, "months": 0}
    )
    assert not bad_value.ok and "months" in bad_value.error


def test_schema_documents_every_argument_and_requires_only_the_essentials():
    params = loan_metrics.definition["function"]["parameters"]
    assert params["required"] == ["principal", "annual_rate_pct", "months"]
    assert set(params["properties"]) == {
        "principal",
        "annual_rate_pct",
        "months",
        "upfront_fee",
        "monthly_income",
        "other_monthly_debts",
        "monthly_payment",
    }
    assert "8.5 means 8.5%" in loan_metrics.description


# --------------------------------------------------------------------------------- Euribor


ECB_REPLY = {
    "dataSets": [{"series": {"0:0:0:0:0:0:0": {"observations": {"0": [2.186, 0, 0, None, None]}}}}],
    "structure": {
        "dimensions": {"observation": [{"id": "TIME_PERIOD", "values": [{"id": "2026-09"}]}]}
    },
}


def test_parse_ecb_observation():
    assert parse_ecb_observation(ECB_REPLY) == ("2026-09", 2.186)


def test_parse_ecb_observation_takes_the_latest_of_several():
    reply = json.loads(json.dumps(ECB_REPLY))
    reply["dataSets"][0]["series"]["0:0:0:0:0:0:0"]["observations"] = {
        "0": [2.1],
        "1": [2.3],
    }
    reply["structure"]["dimensions"]["observation"][0]["values"] = [
        {"id": "2026-08"},
        {"id": "2026-09"},
    ]
    assert parse_ecb_observation(reply) == ("2026-09", 2.3)


@pytest.mark.parametrize("payload", [{}, {"dataSets": []}, {"dataSets": [{"series": {}}]}])
def test_parse_ecb_observation_rejects_empty_replies(payload):
    with pytest.raises(ValueError):
        parse_ecb_observation(payload)


class _FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return ECB_REPLY


def test_get_euribor_calls_the_right_series(monkeypatch):
    seen = {}

    def fake_get(url, **kwargs):
        seen["url"], seen["params"] = url, kwargs["params"]
        return _FakeResponse()

    monkeypatch.setattr(tools.requests, "get", fake_get)
    result = ToolRegistry().register(get_euribor).call("get_euribor", {"tenor": "3M"})
    assert result.ok
    assert result.result == {"tenor": "3M", "period": "2026-09", "rate_pct": 2.186, "source": "ECB"}
    assert "EURIBOR3MD_" in seen["url"] and seen["params"]["lastNObservations"] == 1


def test_get_euribor_default_is_12m_and_invalid_tenor_is_rejected(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda url, **kw: _FakeResponse())
    registry = ToolRegistry().register(get_euribor)
    assert registry.call("get_euribor", {}).result["tenor"] == "12M"
    bad = registry.call("get_euribor", {"tenor": "2Y"})
    assert not bad.ok and "invalid arguments" in bad.error


def test_get_euribor_network_failure_is_an_observation(monkeypatch):
    def boom(url, **kwargs):
        raise tools.requests.ConnectionError("no route")

    monkeypatch.setattr(tools.requests, "get", boom)
    record = ToolRegistry().register(get_euribor).call("get_euribor", {})
    assert not record.ok and "ConnectionError" in record.error


# ------------------------------------------------------------------------- save_credit_report


def test_save_credit_report_writes_a_file_and_requires_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv(tools.REPORTS_DIR_ENV, str(tmp_path))
    assert save_credit_report.requires_confirmation is True
    record = (
        ToolRegistry()
        .register(save_credit_report)
        .call(
            "save_credit_report",
            {
                "applicant_id": "cliente-001",
                "decision": "review",
                "summary": "Cuota 188,20; TAE 11,9%.",
            },
        )
    )
    assert record.ok and record.result["saved"]
    files = list(tmp_path.glob("cliente-001_*.md"))
    assert len(files) == 1
    text = files[0].read_text()
    assert "Decision: review" in text and "TAE 11,9%" in text


@pytest.mark.parametrize("bad_id", ["../etc/passwd", "a/b", "", "x" * 41, "con espacio"])
def test_save_credit_report_rejects_unsafe_ids(tmp_path, monkeypatch, bad_id):
    monkeypatch.setenv(tools.REPORTS_DIR_ENV, str(tmp_path))
    record = (
        ToolRegistry()
        .register(save_credit_report)
        .call(
            "save_credit_report", {"applicant_id": bad_id, "decision": "approve", "summary": "ok"}
        )
    )
    assert not record.ok
    assert list(tmp_path.iterdir()) == []


def test_save_credit_report_rejects_unknown_decisions_and_empty_summaries(tmp_path, monkeypatch):
    monkeypatch.setenv(tools.REPORTS_DIR_ENV, str(tmp_path))
    registry = ToolRegistry().register(save_credit_report)
    assert not registry.call(
        "save_credit_report", {"applicant_id": "a", "decision": "maybe", "summary": "x"}
    ).ok
    assert not registry.call(
        "save_credit_report", {"applicant_id": "a", "decision": "reject", "summary": "   "}
    ).ok
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------- the registry


def test_default_registry_has_the_three_domain_tools():
    registry = default_registry()
    assert set(registry.tools) == {"get_euribor", "loan_metrics", "save_credit_report"}
    confirm = {n for n, t in registry.tools.items() if t.requires_confirmation}
    assert confirm == {"save_credit_report"}
    for definition in registry.definitions():
        assert definition["function"]["description"]


def test_loan_metrics_rejects_an_implausible_rate_and_points_to_the_dependency():
    record = (
        ToolRegistry()
        .register(loan_metrics)
        .call("loan_metrics", {"principal": 10000, "annual_rate_pct": 100.2, "months": 36})
    )
    assert not record.ok
    assert "implausible" in record.error and "call that tool first" in record.error
    assert loan_metrics.fn(principal=10000, annual_rate_pct=60, months=36)["monthly_payment"] > 0