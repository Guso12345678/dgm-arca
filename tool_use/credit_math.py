"""Consumer-credit maths shared by the phase-2 tools and (later) the final project's agents.
 
Pure functions, no I/O, so they can be tested without network or GPU. The conventions are
the ones of the phase-1 dataset (``rlm/data/generar_dataset_credito.py``):
 
* French amortisation: constant monthly instalment, nominal annual rate in percent.
* The instalment is rounded to cents before it is used in anything else, as a bank
  charges it. Total cost and APR are computed from that rounded instalment.
* APR (TAE) solves ``principal - fee = sum_k payment / (1 + i)^k`` for the monthly rate
  ``i`` (bisection) and annualises it: ``(1 + i)^12 - 1``.
"""
 
from __future__ import annotations
 
 
def monthly_payment(principal: float, annual_rate_pct: float, months: int) -> float:
    """French-system instalment, unrounded. ``annual_rate_pct`` is e.g. 8.5, not 0.085."""
    if principal <= 0:
        raise ValueError("principal must be > 0")
    if months < 1:
        raise ValueError("months must be >= 1")
    if annual_rate_pct < 0:
        raise ValueError("annual_rate_pct must be >= 0")
    i = annual_rate_pct / 100 / 12
    if i == 0:
        return principal / months
    return principal * i / (1 - (1 + i) ** (-months))
 
 
def total_cost(payment: float, months: int, upfront_fee: float, principal: float) -> float:
    """Everything paid minus the capital received: interest plus fees."""
    return payment * months + upfront_fee - principal
 
 
def debt_ratio_pct(payment: float, other_monthly_debts: float, monthly_income: float) -> float:
    """Share of net monthly income that goes to debt service, in percent."""
    if monthly_income <= 0:
        raise ValueError("monthly_income must be > 0")
    return (payment + other_monthly_debts) / monthly_income * 100
 
 
def _present_value(payment: float, months: int, i: float) -> float:
    """Present value of ``months`` instalments at monthly rate ``i`` (> 0)."""
    return payment * (1 - (1 + i) ** (-months)) / i
 
 
def apr_pct(principal: float, upfront_fee: float, payment: float, months: int) -> float:
    """APR (TAE) in percent, by bisection on the monthly rate.
 
    Raises ``ValueError`` when the cash flows have no non-negative solution (the borrower
    would repay less than the net amount received).
    """
    net = principal - upfront_fee
    if net <= 0:
        raise ValueError("upfront_fee must be smaller than principal")
    gap_at_zero = payment * months - net  # f(0): sum of instalments minus net amount
    if abs(gap_at_zero) < 1e-9:
        return 0.0
    if gap_at_zero < 0:
        raise ValueError("instalments add up to less than the net amount: no positive APR")
 
    lo, hi = 1e-12, 2.0  # monthly rate between ~0% and 200%
    if _present_value(payment, months, hi) - net > 0:
        raise ValueError("APR out of the supported range")
    for _ in range(200):
        mid = (lo + hi) / 2
        if _present_value(payment, months, mid) - net > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    monthly = (lo + hi) / 2
    return ((1 + monthly) ** 12 - 1) * 100