# forkgate

**A minimal model of axFlow-style fork semantics, not axFlow itself.** This repo is a standalone
Python model written to trial an idea. It does not use axFlow code, and nothing here describes
axFlow's internals or APIs. (axFlow is Chris Taylor's data-forking timestream system.)

## Thesis

1. Humans never eliminated wrongness; institutions contained it.
2. Treat AI outputs the same way: grade every claim, and gate by stakes.
3. Detection of wrongness may arrive days later, so being right at write time is the wrong target.
4. Keep an append-only timestream you can fork at the bad event and replay with the correction.
5. Forking repairs what lives in the stream. It cannot recall an email. That residual is the cost.

## Pipeline

```
 claim --> stakes tier --> gates ---------------------------> outcome
          low / med / high  conformal (self-consistency)      EMIT     all effects allowed
          = balance of      Admiralty reliability x           CONTAIN  provisional; internal ok,
            probabilities /   credibility (separate inputs)             external HELD 3 d
            clear and       ACH stand-in (least refuted)      UNKNOWN  nothing emitted
            convincing /
            beyond
            reasonable doubt
                                        |
                                        v
                       append-only timestream (sha256, parent-linked)
                       internal events | external events (left the system)
                                        |
                  days later: delayed oracle refutes the claim
                                        |
                                        v
                 fork at the claim (prefix EXCLUDES it) --> replay downstream
                      |                                          |
          internal decisions: REPAIRED          external effects: compensation_required
```

## How to run

```
python -m venv .venv && .venv/bin/pip install numpy pytest
.venv/bin/python -m pytest -q          # tests
.venv/bin/python -m forkgate           # the simulation (seeded, deterministic, ~seconds)
.venv/bin/python scenario.py           # one readable case
```

Python 3.11+, stdlib + numpy + pytest. No network calls, no LLM calls.

## Results

Raw output: `results/sim_output.txt`, `results/scenario_output.txt`, `results/test_output.txt`.
Seed 20261102, 10000 claims, parameters frozen in `PREDICTIONS.md` before the first run.

```
                                            |          Arm A |          Arm B |          Arm C |          Arm D
---------------------------------------------------------------------------------------------------------------
wrong claims emitted (EMIT+CONTAIN)         |           3996 |            838 |           1204 |           1204
  of which CONTAINed                        |              0 |              0 |            593 |            593
wrong claims caught (UNKNOWN)               |              0 |           3158 |           2792 |           2792
wrong-confident total                       |            953 |            953 |            953 |            953
wrong-confident caught (UNKNOWN)            |              0 |            259 |            261 |            261
wrong-confident CONTAINed                   |              0 |              0 |            288 |            288
abstention rate                             |           0.0% |          59.5% |          51.8% |          51.8%
correct claims abstained on                 |              0 |           2789 |           2391 |           2391
external holds cancelled (wrong claims)     |              0 |              0 |            202 |            202
contaminated decisions (internal)           |           8459 |           1820 |           2603 |           2603
contaminated decisions (external, executed) |           3542 |            733 |            875 |            875
contaminated repaired by fork (internal)    |              0 |              0 |              0 |           2603
external effects repaired by fork           |              0 |              0 |              0 |              0
external effects needing compensation       | 3542 (unflagged) | 733 (unflagged) | 875 (unflagged) |            875
mean time-to-repair (days)                  |            n/a |            n/a |            n/a |           6.50
```

External effects "needing compensation" exist in every arm; only arm D's replay surfaces them as
`compensation_required` records. In arms A to C they are executed and unflagged.

### Pre-registered predictions

```
Pre-registered predictions:
  1 (B) wrong-confident caught = 259/953 = 27.2%  (need < 20%)  -> FAILED
  2 (C) wrong-confident flagged = 549/953 = 57.6% vs B 27.2%  (need >=40% and >=3x B)  -> FAILED
  3 (D) internal repaired 2603/2603, external repaired 0, compensation_required 875 vs executed external 875  -> PASS
  4 (D) compensation 875 vs A 3542 = 0.25x (need < 0.60)  -> PASS
  5 conformal out-of-sample pass-through <= alpha+0.02 at every tier  -> PASS
```

**Prediction 1: FAILED (27.2% vs the < 20% threshold).** I expected about 10%. My arithmetic
ignored that wrong-confident agreement has a real lower tail (Beta(14,2) on 20 samples), and the
agreement score is quantised in steps of 0.05, so the alpha=0.25 threshold of 0.80 sits inside
that tail. B catches those 259 only by abstaining on 59.5% of all claims, including 2789 correct
ones. So B's catches of wrong-confident claims are a side effect of heavy abstention, not
detection of the misconception.

**Prediction 2: FAILED on its second leg.** 57.6% flagged satisfies the >= 40% leg, but the
>= 3x B leg fails because B's baseline (27.2%) was higher than predicted (3 x 27.2% = 81.6%).
Of C's flags, only 261 are UNKNOWN (about the same as B); the other 288 are CONTAIN, which
holds external effects but still lets internal ones through.

**Prediction 3: PASS.** 2603/2603 internal repaired, 0 external, 875 compensation records.
**Prediction 4: PASS.** 875 vs 3542 (0.25x). Most of the reduction comes from abstention (UNKNOWN
claims spawn nothing); 202 further external effects of wrong claims were held and cancelled.
**Prediction 5: PASS.** Observed pass-through 0.434 / 0.204 / 0.043 against alpha 0.50 / 0.25 / 0.05.

The ACH stand-in overrode 0 claims in this simulation. It is in the pipeline as a stage but does
no measurable work here.

## CHANGES-AFTER-RESULTS

None. No parameter was changed after seeing results. Two dead lines in `forkgate/sim.py`
(a no-op `if it.wrong: pass` and an unused `tampered` counter) were removed after the first run;
a re-run produced byte-identical output.

## Modelling notes

- **Conformal risk** is defined as P(emit | wrong), calibrated on wrong holdout claims with the
  quantile ceil((n+1)(1-alpha))/n. That is my reading of the method, not a claim about what
  arXiv 2405.01563 does. It bounds the share of wrong claims that get through, not the error rate
  among emitted claims.
- **Admiralty** reliability uses track record only (one record per distinct origin; F = no
  record or fewer than 5 observations, which is not E). Credibility uses distinct-origin
  corroboration counts only; sources sharing an origin count once. The mapping from counts to
  grades 1-6 is mine.
- **Stakes tiers**: low alpha 0.50 / D,4; medium 0.25 / C,3; high 0.05 / B,2. These are modelling
  choices; courts do not publish alphas.
- **ACH** here is deterministic arithmetic in three separate functions. The real version must use
  separate model calls (rivals, then one call per hypothesis, with no sight of the others),
  because single-prompt ACH primes the model toward the hypothesis it wrote first. The stand-in
  does not test that failure mode.
- **Streams** are sharded into 100 streams; each correction is its own fix branch off its shard's
  main. No merge back is modelled.

## Limits

- The simulation's populations are synthetic. The numbers show mechanism, not real-world rates.
  In particular, wrong-confident claims were built to have no independent corroboration, which
  is exactly what the Admiralty gate looks for.
- Admiralty grading has no published LLM evaluation. Nothing here shows an LLM can be graded
  this way reliably.
- Consistency oracles detect variance, not bias. A learned misconception that is stable across
  samples passes the conformal gate by construction; the simulation shows this (B above).
- Forking cannot undo external effects. Every email, payment and API call that left the system
  is listed as `compensation_required` and still needs a human or a compensating action.
- `model_confidence` is carried on each claim but no gate uses it.

## References

- Yadkowsky et al., conformal abstention for LLM hallucinations, arXiv 2405.01563
- arXiv 2502.06884
- NATO AJP-2.1 (Admiralty system); Kelly et al. 2025, Judgment and Decision Making
- Heuer, *Psychology of Intelligence Analysis* (Analysis of Competing Hypotheses)
- github.com/suprathermal/ACH-Grounding

## Licence

MIT, copyright Chris Taylor. See `LICENSE`.
