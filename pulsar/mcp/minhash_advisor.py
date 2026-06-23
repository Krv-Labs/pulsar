"""MinHash signature-depth advisory — the speed/accuracy/memory math.

Single source of truth for reasoning about ``cosmic_graph.minhash_d`` (the number of
hash functions). The cosmic graph is built by MinHash: each edge weight is an unbiased
estimate of the Jaccard similarity of two points' ball-sets, averaged over ``d``
independent permutations. The estimator is a mean of ``d`` Bernoulli(J) trials, so its
error depends only on ``d`` — **independent of n or the number of balls**.

Two complementary bounds (both reported so the agent can reason):

- Hoeffding (distribution-free worst case): ``P(|Ŵ−W| ≥ ε) ≤ 2·exp(−2·d·ε²)``
    ⇒ ``ε(d, δ) = sqrt(ln(2/δ) / (2d))`` and ``d(ε, δ) = ceil(ln(2/δ) / (2ε²))``.
- CLT / variance (typical, tighter): ``SE(W) = sqrt(W(1−W)/d)``; worst case at
    W=0.5 → ``SE = 1/(2√d)``; a 95% CI half-width is ``1.96·SE``.

Lowering ``d`` on massive datasets cuts both signature memory (``d·n·4`` bytes) and
construction time (O(d·M), linear in d) at the cost of a wider confidence interval.

References: Broder (1997); Hoeffding (1963); Leskovec, Rajaraman, Ullman, *Mining of
Massive Datasets* ch. 3.
"""

from __future__ import annotations

import math
from typing import Any

# Above this row count, characterization proactively suggests considering a lower `d`.
MASSIVE_N_THRESHOLD = 500_000
DEFAULT_D = 256
DEFAULT_DELTA = 0.05  # 95% confidence
_BYTES_PER_SIGNATURE_ENTRY = 4  # u32 signatures


def epsilon_at_confidence(d: int, delta: float = DEFAULT_DELTA) -> float:
    """Hoeffding worst-case error bound ε such that P(|Ŵ−W| ≥ ε) ≤ delta."""
    if d <= 0:
        raise ValueError("d must be >= 1")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * d))


def depth_for_epsilon(eps: float, delta: float = DEFAULT_DELTA) -> int:
    """Minimum signature depth d guaranteeing Hoeffding error ≤ eps at 1−delta."""
    if not eps > 0.0:
        raise ValueError("eps must be > 0")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    return math.ceil(math.log(2.0 / delta) / (2.0 * eps * eps))


def standard_error(w: float, d: int) -> float:
    """CLT standard error of the Jaccard estimate at true similarity ``w``."""
    if d <= 0:
        raise ValueError("d must be >= 1")
    w = min(max(w, 0.0), 1.0)
    return math.sqrt(w * (1.0 - w) / d)


def ci_half_width(d: int, w: float = 0.5, z: float = 1.96) -> float:
    """Half-width of the ``z``-sigma CI for the Jaccard estimate (default 95% @ W=0.5
    — the worst case)."""
    return z * standard_error(w, d)


def signature_memory_bytes(d: int, n: int) -> int:
    """Bytes for the d×n u32 signature matrix."""
    if d <= 0 or n < 0:
        raise ValueError("d must be >= 1 and n >= 0")
    return d * n * _BYTES_PER_SIGNATURE_ENTRY


def _fmt_bytes(b: int) -> str:
    gb = b / 1e9
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    return f"{b / 1e6:.0f} MB"


def depth_profile(d: int, n: int, delta: float = DEFAULT_DELTA) -> dict[str, Any]:
    """Full accuracy/memory profile for a given depth ``d`` at row count ``n``."""
    return {
        "d": d,
        "hoeffding_epsilon": round(epsilon_at_confidence(d, delta), 4),
        "ci95_half_width_worst": round(ci_half_width(d, 0.5), 4),
        "se_at_0_15": round(standard_error(0.15, d), 4),
        "signature_memory_bytes": signature_memory_bytes(d, n),
        "signature_memory_human": _fmt_bytes(signature_memory_bytes(d, n)),
        "confidence": round(1.0 - delta, 4),
    }


def recommend_d(
    n: int,
    *,
    memory_budget_bytes: int | None = None,
    target_eps: float | None = None,
    candidates: tuple[int, ...] = (512, 256, 128, 64),
    delta: float = DEFAULT_DELTA,
) -> dict[str, Any]:
    """Recommend a signature depth for ``n`` points.

    - With ``target_eps``: smallest standard depth meeting the Hoeffding bound.
    - With ``memory_budget_bytes``: largest standard depth fitting the budget.
    - Otherwise: the default depth, with a note when ``n`` is massive.

    Returns the chosen depth, its accuracy/memory profile, and a human-readable note.
    """
    chosen = DEFAULT_D
    note: str

    if target_eps is not None:
        required = depth_for_epsilon(target_eps, delta)
        # Smallest standard candidate that meets the requirement, else the requirement.
        meeting = sorted(c for c in candidates if c >= required)
        chosen = meeting[0] if meeting else required
        note = (
            f"d={chosen} meets a Hoeffding error of ≤{target_eps:.3f} "
            f"at {int((1 - delta) * 100)}% confidence (requires d≥{required})."
        )
    elif memory_budget_bytes is not None:
        fitting = [
            c for c in candidates if signature_memory_bytes(c, n) <= memory_budget_bytes
        ]
        if fitting:
            chosen = max(fitting)
            note = (
                f"d={chosen} is the largest standard depth whose signature "
                f"({_fmt_bytes(signature_memory_bytes(chosen, n))}) fits the "
                f"{_fmt_bytes(memory_budget_bytes)} budget."
            )
        else:
            chosen = min(candidates)
            note = (
                f"Even d={chosen} needs "
                f"{_fmt_bytes(signature_memory_bytes(chosen, n))}, exceeding the "
                f"{_fmt_bytes(memory_budget_bytes)} budget — consider batching."
            )
    elif n >= MASSIVE_N_THRESHOLD:
        chosen = 128
        default_prof = depth_profile(DEFAULT_D, n, delta)
        chosen_prof = depth_profile(chosen, n, delta)
        note = (
            f"n={n:,} is large: lowering minhash_d {DEFAULT_D}→{chosen} cuts signature "
            f"memory {default_prof['signature_memory_human']}→"
            f"{chosen_prof['signature_memory_human']} and construction time ~2× — "
            f"95% CI widens ±{default_prof['ci95_half_width_worst']}→"
            f"±{chosen_prof['ci95_half_width_worst']}. Default d={DEFAULT_D} is fine "
            f"if memory allows."
        )
    else:
        note = (
            f"Default minhash_d={DEFAULT_D} is appropriate (n={n:,}); no tuning needed."
        )

    return {
        "recommended_d": chosen,
        "profile": depth_profile(chosen, n, delta),
        "note": note,
    }


def massive_dataset_advisory(
    n: int, current_d: int = DEFAULT_D
) -> dict[str, Any] | None:
    """Advisory block for characterization: returned only when ``n`` is massive.

    Quantifies the speed/memory win of lowering ``minhash_d`` and the cost in
    confidence, so the agent can decide. Returns ``None`` for non-massive ``n``.
    """
    if n < MASSIVE_N_THRESHOLD:
        return None
    rec = recommend_d(n)
    return {
        "parameter": "cosmic_graph.minhash_d",
        "current": current_d,
        "current_profile": depth_profile(current_d, n),
        "suggested": rec["recommended_d"],
        "suggested_profile": rec["profile"],
        "message": rec["note"],
    }
