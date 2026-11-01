"""The tools the model can call: consumer-credit risk analysis.

``get_weather`` and ``calculator`` are the worked examples from the slides and are kept
here for reference, but they are NOT registered: the registry the agent sees only has the
three domain tools, one of each type the practice asks for.

* ``get_euribor``  -- queries the outside world (ECB Data API, no key needed).
* ``loan_metrics`` -- computes (instalment, total cost, APR/TAE, debt ratio).
* ``save_credit_report`` -- acts with an observable effect (writes a report file); it asks
  for confirmation (``requires_confirmation=True``).

Design notes (they are also material for the report):

* ``loan_metrics`` is ONE tool that returns every metric instead of four tools. A 1.7B
  model picks the right tool and fills the arguments much more reliably when there is a
  single obvious choice for "any loan calculation"; the model then reads from the result
  the number the user asked for.
* Errors are raised as ``ValueError`` with a message written for the model: the registry
  turns them into observations so the model can correct itself.
* No ``from __future__ import annotations`` here on purpose: the registry builds the
  Pydantic schema from the annotations, and ``Literal`` must be a real object, not a string.
"""

import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import requests

from tool_use import credit_math
from tool_use.registry import ToolRegistry, tool

OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

ECB_DATA_API = "https://data-api.ecb.europa.eu/service/data/FM"
# Monthly average of the Euribor, euro area ("U2"), by maturity.
EURIBOR_SERIES = {
    "1M": "M.U2.EUR.RT.MM.EURIBOR1MD_.HSTA",
    "3M": "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
    "6M": "M.U2.EUR.RT.MM.EURIBOR6MD_.HSTA",
    "12M": "M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA",
}
REPORTS_DIR_ENV = "ARCA_REPORTS_DIR"
DEFAULT_REPORTS_DIR = "reports/credit_reports"
APPLICANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
# Nominal rates above this are almost surely a mistake (e.g. a value guessed before another
# tool returned). Rejecting them turns a silent wrong answer into an observation to retry.
MAX_PLAUSIBLE_RATE_PCT = 60.0


# ---------------------------------------------------------------------------------------
# Examples from the slides (not registered)
# ---------------------------------------------------------------------------------------


@tool
def get_weather(location: str) -> float:
    """Return the current air temperature in degrees Celsius for a location.

    Args:
        location: City and country, for example 'Madrid, Spain'.
    """
    geo = requests.get(OPEN_METEO_GEOCODING, params={"name": location, "count": 1}, timeout=15)
    geo.raise_for_status()
    results = geo.json().get("results") or []
    if not results:
        raise ValueError(f"no coordinates found for {location!r}")
    lat, lon = results[0]["latitude"], results[0]["longitude"]
    forecast = requests.get(
        OPEN_METEO_FORECAST,
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "temperature_unit": "celsius",
        },
        timeout=15,
    )
    forecast.raise_for_status()
    return float(forecast.json()["current_weather"]["temperature"])


@tool
def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression such as '(12.5 * 3) / 2 - 1'.

    Only numbers, parentheses and the operators + - * / ** % are allowed.

    Args:
        expression: The arithmetic expression to evaluate.
    """
    import ast
    import operator

    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.operand))
        raise ValueError("unsupported expression")

    return float(evaluate(ast.parse(expression, mode="eval").body))


# ---------------------------------------------------------------------------------------
# Domain tool 1: outside world (real HTTP API)
# ---------------------------------------------------------------------------------------


def parse_ecb_observation(payload: dict[str, Any]) -> tuple[str, float]:
    """Extract ``(period, value)`` of the latest observation from an ECB ``jsondata`` reply.

    Pure function so it can be tested without network. Raises ``ValueError`` when the reply
    has no observation (unknown series, no data for the period).
    """
    try:
        series = next(iter(payload["dataSets"][0]["series"].values()))
        observations: dict[str, list] = series["observations"]
        last_index = max(observations, key=int)
        value = observations[last_index][0]
        period = payload["structure"]["dimensions"]["observation"][0]["values"][int(last_index)]
    except (KeyError, IndexError, StopIteration, ValueError, TypeError) as exc:
        raise ValueError("the ECB reply has no observation for this series") from exc
    if value is None:
        raise ValueError("the ECB reply has an empty observation")
    return str(period["id"]), float(value)


@tool
def get_euribor(tenor: Literal["1M", "3M", "6M", "12M"] = "12M") -> dict:
    """Return the latest monthly-average Euribor rate, in percent, from the ECB.

    Use it when the user asks for the current Euribor, or for a variable-rate loan whose
    rate is 'Euribor plus a spread'. The Euribor is the rate_pct field of the result: add
    the spread to it yourself afterwards.

    Args:
        tenor: Euribor maturity. '1M' = one month (mensual), '3M' = three months
            (trimestral), '6M' = six months (semestral), '12M' = one year (anual). Mortgages
            and consumer loans usually reference '12M'.
    """
    response = requests.get(
        f"{ECB_DATA_API}/{EURIBOR_SERIES[tenor]}",
        params={"lastNObservations": 1, "format": "jsondata"},
        headers={"Accept": "application/json"},
        timeout=15,
    )
    response.raise_for_status()
    period, rate = parse_ecb_observation(response.json())
    return {
        "tenor": tenor,
        "period": period,
        "rate_pct": round(rate, 3),
        "source": "ECB",
    }


# ---------------------------------------------------------------------------------------
# Domain tool 2: computation
# ---------------------------------------------------------------------------------------


@tool
def loan_metrics(
    principal: float,
    annual_rate_pct: float,
    months: int,
    upfront_fee: float = 0.0,
    monthly_income: float | None = None,
    other_monthly_debts: float = 0.0,
    monthly_payment: float | None = None,
) -> dict:
    """Compute the figures of a consumer loan with French amortisation (constant instalment).

    Returns monthly_payment (euros), total_cost (euros: interest plus fees, i.e. everything
    paid minus the capital received), apr_pct (TAE, in percent) and debt_ratio_pct (percent
    of net income that goes to debt, or null if monthly_income is not given). All values are
    rounded to 2 decimals. Always use this tool for these calculations instead of computing
    them by hand.

    Args:
        principal: Loan amount in euros, for example 12000.
        annual_rate_pct: Nominal annual interest rate in percent: 8.5 means 8.5%, not 0.085.
            For 'Euribor plus a spread' it is the Euribor returned by get_euribor plus the
            spread, already added up; never guess the Euribor.
        months: Term of the loan in months.
        upfront_fee: Opening fee in euros, paid at the start. Defaults to 0.
        monthly_income: Net monthly income in euros. Only needed for the debt ratio.
        other_monthly_debts: Other monthly debt payments in euros. Defaults to 0.
        monthly_payment: Instalment in euros if it is already known; if omitted it is
            computed from the principal, the rate and the term.
    """
    if annual_rate_pct > MAX_PLAUSIBLE_RATE_PCT:
        raise ValueError(
            f"annual_rate_pct={annual_rate_pct:g} is implausible (expected 0 to "
            f"{MAX_PLAUSIBLE_RATE_PCT:g}, in percent: 8.5 means 8.5%). If the rate depends on "
            "another tool's result (for example the Euribor), call that tool first, wait for "
            "its result, and then call loan_metrics again."
        )
    if monthly_payment is None:
        payment = round(credit_math.monthly_payment(principal, annual_rate_pct, months), 2)
    else:
        if monthly_payment <= 0:
            raise ValueError("monthly_payment must be > 0")
        payment = round(monthly_payment, 2)

    result: dict[str, Any] = {
        "monthly_payment": payment,
        "total_cost": round(credit_math.total_cost(payment, months, upfront_fee, principal), 2),
        "apr_pct": round(credit_math.apr_pct(principal, upfront_fee, payment, months), 2),
        "debt_ratio_pct": None,
    }
    if monthly_income is not None:
        ratio = credit_math.debt_ratio_pct(payment, other_monthly_debts, monthly_income)
        result["debt_ratio_pct"] = round(ratio, 2)
    return result


# ---------------------------------------------------------------------------------------
# Domain tool 3: action with an observable effect
# ---------------------------------------------------------------------------------------


@tool(requires_confirmation=True)
def save_credit_report(
    applicant_id: str, decision: Literal["approve", "review", "reject"], summary: str
) -> dict:
    """Save a credit-risk report for an applicant as a markdown file. It has side effects.

    Only call it when the user explicitly asks to save or file the report, and after the
    figures it should contain have been computed.

    Args:
        applicant_id: Identifier of the applicant: letters, digits, '-' or '_' (max 40).
        decision: Recommended decision: 'approve', 'review' (manual analysis) or 'reject'.
        summary: The justification: the key figures (instalment, APR, debt ratio) and why.
    """
    if not APPLICANT_ID_PATTERN.match(applicant_id):
        raise ValueError("applicant_id may only contain letters, digits, '-' and '_' (max 40)")
    if not summary.strip():
        raise ValueError("summary must not be empty")

    directory = Path(os.environ.get(REPORTS_DIR_ENV, DEFAULT_REPORTS_DIR))
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    path = directory / f"{applicant_id}_{now:%Y%m%dT%H%M%S%f}.md"
    path.write_text(
        f"# Credit report: {applicant_id}\n\n"
        f"- Date (UTC): {now:%Y-%m-%d %H:%M:%S}\n"
        f"- Decision: {decision}\n\n"
        f"{summary.strip()}\n",
        encoding="utf-8",
    )
    return {"saved": True, "path": str(path), "decision": decision}


def default_registry() -> ToolRegistry:
    """The registry the executor and the agent use."""
    return ToolRegistry().register(get_euribor, loan_metrics, save_credit_report)