"""Canary promotion: error rate + latency from live traffic, plus a small exact-answer eval.

Traffic phase: ``cortex-default`` splits 90/10 between the stable deployment and the candidate.
Each outcome records which deployment answered (``x-litellm-model-id``), so per-version error rate
and TTFT come from the same traffic. Eval phase: the same labelled probe set is sent to both
versions through pinned aliases (``cortex-stable`` / ``cortex-candidate``) and scored by exact match.

The decision is a pure function of those numbers (``decide``), so the rule is unit-tested and the
same inputs always give the same verdict. Rule (all must hold to promote):

1. enough evidence: >= ``min_canary_requests`` candidate requests in the traffic window;
2. errors: candidate error rate <= max(``max_error_rate``, stable error rate);
3. latency: candidate TTFT p95 <= ``max_ttft_ratio`` x stable TTFT p95;
4. quality: candidate eval accuracy >= stable accuracy - ``max_accuracy_drop``.

Too little evidence gives HOLD (keep 10%, collect more); any failed check gives ROLLBACK.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from cortex_serve.gatewaybench.load import Outcome
from cortex_serve.servebench.metrics import percentile

Verdict = Literal["PROMOTE", "HOLD", "ROLLBACK"]


@dataclass(frozen=True)
class Policy:
    min_canary_requests: int = 30
    max_error_rate: float = 0.01
    max_ttft_ratio: float = 1.25
    max_accuracy_drop: float = 0.05


@dataclass
class VersionStats:
    requests: int
    errors: int
    ttft_p95_s: float | None

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0


@dataclass
class Decision:
    verdict: Verdict
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def traffic_stats(outcomes: Sequence[Outcome], deployment_substring: str) -> VersionStats:
    """Stats for outcomes answered by deployments whose id contains ``deployment_substring``.

    Failed requests may carry no deployment header (the gateway never reached a backend); those are
    attributed to neither version here and show up in the overall error count instead.
    """
    mine = [o for o in outcomes if o.deployment and deployment_substring in o.deployment]
    ttfts = [o.ttft_s for o in mine if o.ok and o.ttft_s is not None]
    return VersionStats(requests=len(mine), errors=sum(not o.ok for o in mine), ttft_p95_s=percentile(ttfts, 95))


def decide(
    stable: VersionStats, canary: VersionStats, stable_acc: float, canary_acc: float, policy: Policy
) -> Decision:
    d = Decision(verdict="PROMOTE")
    if canary.requests < policy.min_canary_requests:
        d.verdict = "HOLD"
        d.reasons.append(f"only {canary.requests} canary requests (< {policy.min_canary_requests})")
        return d
    err_limit = max(policy.max_error_rate, stable.error_rate)
    d.checks["errors"] = canary.error_rate <= err_limit
    if not d.checks["errors"]:
        d.reasons.append(f"canary error rate {canary.error_rate:.2%} > {err_limit:.2%}")
    if canary.ttft_p95_s is None or stable.ttft_p95_s is None:
        d.checks["latency"] = False
        d.reasons.append("no TTFT p95 for one of the versions")
    else:
        limit = policy.max_ttft_ratio * stable.ttft_p95_s
        d.checks["latency"] = canary.ttft_p95_s <= limit
        if not d.checks["latency"]:
            d.reasons.append(
                f"canary TTFT p95 {canary.ttft_p95_s:.2f} s > {policy.max_ttft_ratio} x stable = {limit:.2f} s"
            )
    d.checks["quality"] = canary_acc >= stable_acc - policy.max_accuracy_drop
    if not d.checks["quality"]:
        d.reasons.append(f"canary accuracy {canary_acc:.0%} < stable {stable_acc:.0%} - {policy.max_accuracy_drop:.0%}")
    if not all(d.checks.values()):
        d.verdict = "ROLLBACK"
    return d


# Exact-answer probe set: short, unambiguous, checkable by string match. Small on purpose (CPU box);
# it gates gross regressions (a broken quant, wrong chat template), not fine quality differences.
PROBES: list[tuple[str, str]] = [
    ("What is 7 + 5? Answer with the number only.", "12"),
    ("What is 9 times 8? Answer with the number only.", "72"),
    ("What is 100 minus 37? Answer with the number only.", "63"),
    ("What is 15 divided by 3? Answer with the number only.", "5"),
    ("What is 6 squared? Answer with the number only.", "36"),
    ("What is the capital of France? Answer with one word.", "paris"),
    ("What is the capital of Japan? Answer with one word.", "tokyo"),
    ("What is the capital of Italy? Answer with one word.", "rome"),
    ("What is the capital of Germany? Answer with one word.", "berlin"),
    ("What is the capital of Spain? Answer with one word.", "madrid"),
    ("What color is the sky on a clear day? Answer with one word.", "blue"),
    ("How many days are in a week? Answer with the number only.", "7"),
    ("How many legs does a spider have? Answer with the number only.", "8"),
    ("What is the chemical symbol for gold? Answer with the symbol only.", "au"),
    ("What is the chemical symbol for water? Answer with the formula only.", "h2o"),
    ("What is the opposite of 'hot'? Answer with one word.", "cold"),
    ("Which planet is known as the Red Planet? Answer with one word.", "mars"),
    ("What is the largest ocean on Earth? Answer with one word.", "pacific"),
    ("How many hours are in a day? Answer with the number only.", "24"),
    ("What is 3 + 4 * 2? Answer with the number only.", "11"),
]


def normalize(text: str) -> str:
    """Lowercase, drop a leading <think>...</think> block, keep alphanumerics."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def is_correct(answer: str, expected: str) -> bool:
    """Exact match on the first word, or the expected token anywhere in a short answer."""
    norm = normalize(answer)
    words = norm.split()
    if not words:
        return False
    return words[0] == expected or (len(words) <= 6 and expected in words)
