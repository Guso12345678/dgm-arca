"""The with/without-tools comparison: number parsing, scoring and aggregation (no GPU)."""

import json
from types import SimpleNamespace

import pytest

from tool_use import compare


@pytest.mark.parametrize(
    ("token", "value"),
    [
        ("378.81", 378.81),
        ("378,81", 378.81),
        ("1,948.25", 1948.25),
        ("1.948,25", 1948.25),
        ("26,584.56", 26584.56),
        ("1948", 1948.0),
        ("12.345.678", 12345678.0),
        ("1,948,000", 1948000.0),
        ("1,948", 1948.0),
        ("0.50", 0.5),
    ],
)
def test_parse_number(token, value):
    assert compare.parse_number(token) == pytest.approx(value)


def test_parse_number_rejects_non_numbers():
    assert compare.parse_number("...") is None
    assert compare.parse_number("") is None


def test_is_correct_accepts_the_value_in_free_text_and_markdown():
    assert compare.is_correct("La cuota mensual es de **378.81 euros**.", "378.81")
    assert compare.is_correct("La cuota sería de 378,81 euros al mes", "378.81")
    assert compare.is_correct("<answer>1948.25</answer>", "1948.25")
    assert compare.is_correct(
        "Total pagado 26,803.97; coste del crédito: 1,948.25 euros", "1948.25"
    )


def test_is_correct_is_strict_about_the_value():
    assert not compare.is_correct("La cuota es de 378.90 euros.", "378.81")
    assert not compare.is_correct("No lo sé", "378.81")
    assert compare.is_correct("378.814", "378.81")  # within half a cent


def test_numbers_inside_think_blocks_do_not_count():
    answer = "<think>quizá 378.81</think>La cuota es 400.00 euros."
    assert not compare.is_correct(answer, "378.81")
    assert compare.is_correct(answer, "400.00")


def write_dataset(path, per_type=4):
    rows = []
    for tipo in ("cuota", "tae"):
        for i in range(per_type):
            rows.append(
                {"id": f"{tipo}_{i}", "tipo": tipo, "question": f"q {tipo} {i}", "answer": "1.00"}
            )
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_load_problems_samples_per_type_and_is_reproducible(tmp_path):
    path = tmp_path / "test.jsonl"
    write_dataset(path)
    first = compare.load_problems(path, per_type=2, seed=1)
    again = compare.load_problems(path, per_type=2, seed=1)
    assert [p["id"] for p in first] == [p["id"] for p in again]
    assert sorted(p["tipo"] for p in first) == ["cuota", "cuota", "tae", "tae"]
    assert len(compare.load_problems(path, per_type=None)) == 8
    assert len(compare.load_problems(path, per_type=99)) == 8


class FakeLoop:
    """Without tools the 'model' guesses 0.00; with tools it answers the right value."""

    def generate(self, messages):
        assert [m["role"] for m in messages] == ["user"]  # no tool system prompt
        return "Creo que son 0.00 euros."

    def run(self, query, max_turns=5):
        call = SimpleNamespace(name="loan_metrics", arguments={"principal": 1}, ok=True)
        return SimpleNamespace(answer="El resultado es 1.00 euros.", tool_calls=[call], turns=2)


def test_evaluate_and_summarise_with_a_fake_loop(tmp_path):
    path = tmp_path / "test.jsonl"
    write_dataset(path)
    problems = compare.load_problems(path, per_type=None)
    rows = compare.evaluate(FakeLoop(), problems)
    assert all(not r["without_tools"]["correct"] and r["with_tools"]["correct"] for r in rows)
    assert rows[0]["with_tools"]["tool_calls"][0]["name"] == "loan_metrics"

    summary = compare.summarise(rows)
    assert set(summary) == {"cuota", "tae", "TOTAL"}
    assert summary["TOTAL"] == {
        "n": 8,
        "without_tools": 0.0,
        "with_tools": 1.0,
        "used_loan_metrics": 1.0,
    }


def test_summary_counts_only_successful_loan_metrics_calls():
    failed = SimpleNamespace(name="loan_metrics", ok=False)
    rows = [
        {
            "tipo": "tae",
            "without_tools": {"correct": False},
            "with_tools": {
                "correct": False,
                "tool_calls": [{"name": failed.name, "ok": failed.ok}],
            },
        }
    ]
    assert compare.summarise(rows)["tae"]["used_loan_metrics"] == 0.0


def test_table_prints_every_group(capsys):
    compare.print_table(
        {
            "tae": {"n": 2, "without_tools": 0.0, "with_tools": 1.0, "used_loan_metrics": 1.0},
            "TOTAL": {"n": 2, "without_tools": 0.0, "with_tools": 1.0, "used_loan_metrics": 1.0},
        }
    )
    out = capsys.readouterr().out
    assert "tae" in out and out.index("tae") < out.index("TOTAL")