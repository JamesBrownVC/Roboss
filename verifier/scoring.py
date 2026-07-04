"""Turn a violation list into a plausibility score and accept/reject decision.

score = 1 - sum(weight[type] * worst_severity_of_that_type)

Only the worst violation of each type is counted, so ten small foot skates
don't outweigh one teleporting box.
"""

from __future__ import annotations

from .config import WEIGHTS, Thresholds
from .tracks import Violation


def plausibility_score(violations: list[Violation]) -> float:
    worst: dict[str, float] = {}
    for v in violations:
        worst[v.type] = max(worst.get(v.type, 0.0), v.severity)
    score = 1.0 - sum(WEIGHTS.get(t, 0.10) * s for t, s in worst.items())
    return max(0.0, min(1.0, score))


def decide(violations: list[Violation], th: Thresholds) -> tuple[bool, float, str]:
    """Returns (plausible, score, main_reason)."""
    score = plausibility_score(violations)
    critical = [v for v in violations if v.severity > th.critical_severity]
    plausible = score >= th.accept_score and not critical
    if plausible:
        reason = "No critical physical inconsistencies detected."
    elif critical:
        reason = (f"Critical violation: {critical[0].type} "
                  f"(severity {critical[0].severity:.2f}) — {critical[0].reason}")
    else:
        types = sorted({v.type for v in violations})
        reason = ("The generated video contains physical inconsistencies: "
                  + ", ".join(t.replace("_", " ") for t in types) + ".")
    return plausible, score, reason
