# Pre-registered predictions (FORKGATE-1)

Written and committed BEFORE the simulation was run for the first time.
Parameters below are frozen. Any change made after seeing results must be
recorded in README.md under CHANGES-AFTER-RESULTS.

## Frozen simulation parameters

- Seed 20261102, N = 10000 claims, calibration holdout = 5000 claims (separate draw).
- Populations: correct-confident 40%, wrong-uncertain 30%, wrong-confident 10%, correct-uncertain 20%.
  (Wrong total = 40%; wrong-confident = 25% of all wrong claims. This mix was chosen by
  back-of-envelope arithmetic before any run, see prediction 1's reasoning. It was not tuned on output.)
- Stakes mix: low 50%, medium 35%, high 15%, independent of population.
- Conformal risk = P(claim emitted | claim wrong), calibrated on wrong holdout claims.
  Tier alpha: low 0.50, medium 0.25, high 0.05. Arm B uses a single alpha = 0.25 for every claim.
- Admiralty minimums (reliability, credibility): low (D, 4), medium (C, 3), high (B, 2).
- Gate failure: conformal fail -> UNKNOWN; ACH override -> UNKNOWN; Admiralty fail -> CONTAIN
  (low/medium) or UNKNOWN (high); else EMIT.
- CONTAIN holds external effects for 3 simulated days; a hold is cancelled if the claim is
  refuted before release (arms C and D only).
- Oracle delay: lognormal, median 4 days, sigma 1.0, clipped to [0.25, 90] days. Every wrong
  emitted claim is eventually refuted inside the run horizon.

## Predictions

1. **(B) conformal only catches almost no wrong-confident claims.**
   Reasoning: wrong-confident claims have the same self-consistency distribution as
   correct-confident ones. Wrong-confident is 25% of wrongs, and the guarantee is aggregate
   (pass-through <= 0.25), so the threshold can sit above the wrong-uncertain mass and below
   the wrong-confident mass.
   Pass criterion: B catches (abstains on) < 20% of wrong-confident claims. Expected about 10%.
2. **(C) conformal + Admiralty + stakes tiers flags materially more wrong-confident claims**
   because they lack independent corroboration.
   Pass criterion: share of wrong-confident claims flagged (UNKNOWN or CONTAIN) >= 40% AND
   >= 3x B's caught share. Expected about 50%: low-stakes claims (50% of the mix) are
   deliberately allowed through by the low tier.
3. **(D) fork/replay repairs all internal contamination and zero external.**
   Pass criterion: repaired internal decisions == contaminated internal decisions (100%),
   external effects repaired == 0, and compensation_required count == executed external effects
   of wrong emitted claims, and that count is > 0. Compensation is the residual cost.
4. **(D) compensation count is below arm A's** because abstention and held-then-cancelled
   external effects never leave the system. Pass criterion: D compensation < 0.6 x A compensation.
5. **Conformal guarantee holds out of sample.** Pass criterion: on a fresh draw, wrong-claim
   pass-through <= alpha + 0.02 at each tier alpha.
