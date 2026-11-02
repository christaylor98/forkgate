"""Oracle gates. Each is a pure function over a Claim.

  a. conformal abstention  (split conformal on self-consistency agreement, arXiv 2405.01563)
  b. Admiralty dual grading (NATO AJP-2.1: reliability A-F, credibility 1-6, computed separately)
  c. ACH stand-in           (Heuer; deterministic, three separate steps, no LLM)
  d. stakes tiers           (court standards pick alpha and the minimum Admiralty grade)

Modelling choices that are mine, not the references': the definition of risk in (a), the
credibility mapping in (b), the ACH evidence weights in (c), and the numbers in (d).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

EMIT, UNKNOWN, CONTAIN = "EMIT", "UNKNOWN", "CONTAIN"


@dataclass(frozen=True)
class Source:
    origin: str                                      # who originally said it (not who repeated it)
    track_record: Optional[tuple[int, int]] = None   # (hits, total) of this origin; None = no record
    supports: bool = True


@dataclass(frozen=True)
class Claim:
    text: str
    claim_type: str
    stakes: str                                      # "low" | "medium" | "high"
    sources: tuple[Source, ...]
    model_confidence: float                          # recorded; deliberately NOT used by any gate
    samples: tuple[str, ...]                         # k resampled answers


# ---------------------------------------------------------------- (a) conformal
def agreement(samples: Sequence[str]) -> float:
    """Self-consistency: fraction of samples that match the modal answer."""
    return max(Counter(samples).values()) / len(samples) if samples else 0.0


def asserted_answer(samples: Sequence[str]) -> str:
    c = Counter(samples)
    top = max(c.values())
    return next(s for s in samples if c[s] == top)   # first-seen among ties: deterministic


def conformal_threshold(wrong_scores: Sequence[float], alpha: float) -> float:
    """Split-conformal threshold. Calibration scores are the agreement of claims known to be
    WRONG. With n scores, take the k-th smallest where k = ceil((n+1)(1-alpha)), i.e. the
    empirical quantile at level ceil((n+1)(1-alpha))/n. A new wrong claim is exchangeable with
    them, so P(agreement > threshold | wrong) <= alpha. If k > n there is no valid threshold
    and everything abstains (+inf).

    Risk here means P(emit | wrong). It is NOT the error rate among emitted claims."""
    s = sorted(wrong_scores)
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else s[k - 1]


def conformal_abstain(claim: Claim, tau: float) -> bool:
    return agreement(claim.samples) <= tau


# ---------------------------------------------------------------- (b) Admiralty
MIN_TRACK = 5
REL_ORDER = "ABCDE"        # F sits outside the order: unjudged, not worse than E


def reliability_grade(track_records: Sequence[Optional[tuple[int, int]]]) -> str:
    """Source reliability A-F from track record ONLY (one record per distinct origin).
    F = cannot be judged (no record, or too little history). E = judged, and bad."""
    best = None
    for rec in track_records:
        if rec is None or rec[1] < MIN_TRACK:
            continue
        rate = rec[0] / rec[1]
        g = ("A" if rate >= .95 else "B" if rate >= .85 else "C" if rate >= .70
             else "D" if rate >= .50 else "E")
        if best is None or REL_ORDER.index(g) < REL_ORDER.index(best):
            best = g
    return best or "F"


def credibility_grade(n_supporting_origins: int, n_contradicting_origins: int) -> int:
    """Information credibility 1-6 from corroboration ONLY. Origins, not citations:
    callers must have collapsed shared-origin sources already (see origin_counts)."""
    s, c = n_supporting_origins, n_contradicting_origins
    if c > 0 and c >= s:
        return 5                      # improbable
    if s >= 4:
        return 1                      # confirmed by other independent sources
    if s == 3:
        return 2
    if s == 2:
        return 3
    if s == 1:
        return 4                      # one origin, uncorroborated
    return 6                          # nothing to judge from


def reliability_inputs(sources: Sequence[Source]) -> list[Optional[tuple[int, int]]]:
    """One track record per distinct origin (the first one seen)."""
    seen: dict[str, Optional[tuple[int, int]]] = {}
    for s in sources:
        seen.setdefault(s.origin, s.track_record)
    return list(seen.values())


def origin_counts(sources: Sequence[Source]) -> tuple[int, int]:
    """(supporting, contradicting) DISTINCT origins. Ten pages citing one origin = one source."""
    sup = {s.origin for s in sources if s.supports}
    con = {s.origin for s in sources if not s.supports}
    return len(sup), len(con)


def admiralty(claim: Claim) -> tuple[str, int]:
    """(reliability, credibility). The two functions never see each other's inputs."""
    return (reliability_grade(reliability_inputs(claim.sources)),
            credibility_grade(*origin_counts(claim.sources)))


# ---------------------------------------------------------------- (d) stakes tiers
@dataclass(frozen=True)
class Tier:
    name: str
    standard: str
    alpha: float          # P(emit | wrong) target for the conformal gate
    min_reliability: str  # letter, best A
    max_credibility: int  # number, best 1


TIERS: Mapping[str, Tier] = {
    "low": Tier("low", "balance of probabilities", 0.50, "D", 4),
    "medium": Tier("medium", "clear and convincing", 0.25, "C", 3),
    "high": Tier("high", "beyond reasonable doubt", 0.05, "B", 2),
}


def admiralty_ok(rel: str, cred: int, tier: Tier) -> bool:
    if rel == "F":                  # unjudged source never meets a minimum
        return False
    return REL_ORDER.index(rel) <= REL_ORDER.index(tier.min_reliability) and cred <= tier.max_credibility


# ---------------------------------------------------------------- (c) ACH stand-in
# The real version must run each step as a SEPARATE model call: one call to generate rivals,
# then one call PER hypothesis to evaluate evidence against it, with no sight of the other
# hypotheses. Single-prompt ACH primes the model toward the hypothesis it wrote first.
# The stand-in below is deterministic arithmetic and is structurally separate but not a test
# of that failure mode.
def ach_generate_hypotheses(claim: Claim, max_rivals: int = 2) -> list[str]:
    asserted = asserted_answer(claim.samples)
    rivals = [a for a, _ in Counter(claim.samples).most_common() if a != asserted]
    return [asserted] + rivals[:max_rivals]


def ach_evaluate(claim: Claim, hypothesis: str) -> float:
    """Refutation weight for ONE hypothesis. Lower is better supported. Sees no other hypothesis."""
    n = len(claim.samples)
    asserted = asserted_answer(claim.samples)
    sup, con = origin_counts(claim.sources)
    refuted = sum(1 for a in claim.samples if a != hypothesis) / n
    if hypothesis == asserted:
        refuted += 0.25 * con         # independent origins contradicting the assertion
    else:
        refuted += 0.25 * sup         # independent origins supporting the assertion refute rivals
    return refuted


def ach_select(scores: Mapping[str, float]) -> str:
    """Least refuted wins (not most supported). Ties go to the first hypothesis listed."""
    return min(scores, key=lambda h: scores[h])


def ach(claim: Claim) -> str:
    hyps = ach_generate_hypotheses(claim)
    return ach_select({h: ach_evaluate(claim, h) for h in hyps})


# ---------------------------------------------------------------- decision
@dataclass(frozen=True)
class Decision:
    outcome: str
    reasons: tuple[str, ...] = field(default_factory=tuple)


def decide_conformal_only(claim: Claim, tau: float) -> Decision:
    if conformal_abstain(claim, tau):
        return Decision(UNKNOWN, ("conformal",))
    return Decision(EMIT)


def decide(claim: Claim, taus: Mapping[str, float]) -> Decision:
    """Arm C/D gate. taus maps tier name -> calibrated conformal threshold.
    UNKNOWN: nothing emitted. CONTAIN: emitted as provisional, internal effects allowed,
    external effects held. EMIT: no restriction."""
    tier = TIERS[claim.stakes]
    if conformal_abstain(claim, taus[tier.name]):
        return Decision(UNKNOWN, ("conformal",))
    if ach(claim) != asserted_answer(claim.samples):
        return Decision(UNKNOWN, ("ach",))
    rel, cred = admiralty(claim)
    if not admiralty_ok(rel, cred, tier):
        reason = (f"admiralty {rel}{cred} below {tier.min_reliability}{tier.max_credibility}",)
        return Decision(UNKNOWN if tier.name == "high" else CONTAIN, reason)
    return Decision(EMIT)
