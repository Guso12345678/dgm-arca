"""Verifier interface for reinforcement learning with verifiable rewards.

A verifier answers one question deterministically: *is this answer correct for
this problem?* Everything in phase 1 hangs on it. If the verifier is sloppy, the
model will learn to exploit the sloppiness instead of learning to reason.

We ship two verifiers:

* ``NumericVerifier`` compares numbers after normalisation. It is what GSM8K needs
  and what the smoke test uses.
* ``ExactMatchVerifier`` compares normalised strings. Useful for multiple-choice
  or short factual answers.

Your domain verifier goes in this module too. Subclass ``Verifier``, implement
``is_correct`` and add a test for it in ``tests/test_verifier.py``. Common shapes:
run unit tests on generated code, execute a SQL query and compare result sets,
validate a JSON document against a schema, check that a date falls in a range.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rlm.rewards import extract_answer, normalize_number


@dataclass(frozen=True)
class VerificationResult:
    """What a verifier reports back. ``detail`` is free text for logging and debugging."""

    is_correct: bool
    predicted: str | None
    expected: str
    detail: str = ""


class Verifier(ABC):
    """Base class for all verifiers."""

    name: str = "verifier"

    @abstractmethod
    def is_correct(self, predicted: str | None, expected: str) -> bool:
        """Return True when ``predicted`` should be accepted as a correct answer."""

    def verify(self, completion: str, expected: str) -> VerificationResult:
        """Extract the final answer from a full completion and check it."""
        predicted = extract_answer(completion)
        ok = self.is_correct(predicted, expected)
        detail = "no <answer> block found" if predicted is None else ""
        return VerificationResult(ok, predicted, expected, detail)


class NumericVerifier(Verifier):
    """Numeric comparison with an optional absolute tolerance."""

    name = "numeric"

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = tolerance

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        p, e = normalize_number(predicted), normalize_number(expected)
        if p is None or e is None:
            return False
        if self.tolerance == 0.0:
            return p == e
        return abs(float(p) - float(e)) <= self.tolerance


class ExactMatchVerifier(Verifier):
    """Case- and whitespace-insensitive string comparison."""

    name = "exact_match"

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        return self._normalize(predicted) == self._normalize(expected)


"""Verifica respuestas numéricas de los problemas de crédito al consumo
    (cuota francesa, ratio de endeudamiento, coste total del crédito, TAE).

    Los cuatro tipos se verifican igual: se normaliza el número extraído de
    la respuesta del modelo y el guardado como correcto, se redondean ambos
    a `decimals` posiciones decimales y se comparan como strings."""
class CreditoVerifier(Verifier): 
    name = "credito"

    def __init__(self, decimals= 2):
        self.decimals = decimals

    def _round(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_number(value)
        if normalized is None:
            return None
        rounded = round(float(normalized), self.decimals)
        rounded = 0.0 if rounded == 0 else rounded
        return f"{rounded:.{self.decimals}f}"

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        predicted_rounded = self._round(predicted)
        expected_rounded = self._round(expected)
        if predicted_rounded is None or expected_rounded is None:
            return False
        return predicted_rounded == expected_rounded
