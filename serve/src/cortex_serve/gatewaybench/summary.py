"""Outcome-log summary: totals, failures, which deployment served what, retries and fallbacks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from cortex_serve.gatewaybench.load import Outcome
from cortex_serve.servebench.metrics import percentile


def summarize_outcomes(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    ok = [o for o in outcomes if o.ok]
    ttft = [o.ttft_s for o in ok if o.ttft_s is not None]
    e2e = [o.e2e_s for o in ok if o.e2e_s is not None]
    errors = Counter((o.error or f"status {o.status}")[:120] for o in outcomes if not o.ok)
    starts = [o.start_unix for o in outcomes]
    return {
        "requests": len(outcomes),
        "ok": len(ok),
        "failed": len(outcomes) - len(ok),
        "error_rate": (len(outcomes) - len(ok)) / len(outcomes) if outcomes else None,
        "errors": dict(errors.most_common()),
        "by_deployment": dict(Counter(o.deployment or "none" for o in outcomes).most_common()),
        "ok_by_deployment": dict(Counter(o.deployment or "none" for o in ok).most_common()),
        "with_retries": sum(1 for o in outcomes if (o.attempted_retries or 0) > 0),
        "with_fallbacks": sum(1 for o in outcomes if (o.attempted_fallbacks or 0) > 0),
        "window_s": (max(starts) - min(starts)) if starts else None,
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_p95_s": percentile(ttft, 95),
        "e2e_p50_s": percentile(e2e, 50),
        "e2e_p95_s": percentile(e2e, 95),
        "output_tokens_total": sum(o.output_tokens or 0 for o in ok),
    }
