"""Seeded simulation: four arms over 10000 synthetic claims. No network, no LLM.

Truth is generated alongside each claim but is never part of the Claim the gates see.
All numbers describe the mechanism on synthetic populations, not real-world rates.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

from . import gates as G
from .timestream import EXTERNAL, INTERNAL, Timestream

SEED = 20261102
N_CLAIMS = 10_000
N_CALIB = 5_000
K_SAMPLES = 20
N_SHARDS = 100
HOLD_DAYS = 3.0
ARM_B_ALPHA = 0.25
ARRIVAL_SPAN_DAYS = 120.0

POPS = ["correct_confident", "wrong_uncertain", "wrong_confident", "correct_uncertain"]
POP_P = [0.40, 0.30, 0.10, 0.20]
STAKES = ["low", "medium", "high"]
STAKES_P = [0.50, 0.35, 0.15]
# self-consistency ~ Beta(a, b). wrong_confident deliberately matches correct_confident.
CONSISTENCY = {"correct_confident": (14, 2), "wrong_confident": (14, 2),
               "wrong_uncertain": (4, 4), "correct_uncertain": (5, 3)}
TRACK = {"correct_confident": (9, 1.5), "correct_uncertain": (7, 2),
         "wrong_uncertain": (5, 3), "wrong_confident": (8, 2)}


@dataclass
class Item:
    claim: G.Claim
    pop: str
    wrong: bool
    t: float          # arrival (days)
    delay: float      # oracle delay; only meaningful if wrong
    n_decisions: int
    ext_flags: list


def _track_record(rng, pop):
    if rng.random() < 0.10:
        return None if rng.random() < .5 else (int(rng.integers(0, 4)), int(rng.integers(1, 5)))
    total = int(rng.integers(8, 120))
    return (int(rng.binomial(total, rng.beta(*TRACK[pop]))), total)


def _sources(rng, pop, cid):
    sources = []
    if pop == "wrong_confident":
        # learned misconception: many citations, one origin, rarely any independent corroboration
        n_indep = 1 if rng.random() < 0.12 else 0
        echoes = int(rng.integers(1, 5)) if rng.random() < 0.6 else 0
        rec = _track_record(rng, pop)
        for j in range(echoes + 1):
            sources.append(G.Source(f"echo-origin-{cid}", rec))
    else:
        if pop == "correct_confident":
            n_indep = int(rng.choice([2, 3, 4], p=[.4, .4, .2])) if rng.random() < .85 else int(rng.integers(0, 2))
        elif pop == "correct_uncertain":
            n_indep = int(rng.choice([1, 2, 3])) if rng.random() < .6 else int(rng.integers(0, 2))
        else:
            n_indep = int(rng.choice([0, 1, 2], p=[.4, .4, .2]))
    for j in range(n_indep):
        sources.append(G.Source(f"indep-{cid}-{j}", _track_record(rng, pop)))
    contradict = (rng.random() < .10) if pop.startswith("wrong") else (rng.random() < .03)
    if contradict:
        sources.append(G.Source(f"contra-{cid}", _track_record(rng, "correct_confident"), supports=False))
    return tuple(sources)


def gen_claim(rng, cid):
    pop = POPS[int(rng.choice(4, p=POP_P))]
    stakes = STAKES[int(rng.choice(3, p=STAKES_P))]
    p = float(rng.beta(*CONSISTENCY[pop]))
    samples = tuple("ans-0" if rng.random() < p else f"alt-{int(rng.integers(1, 5))}"
                    for _ in range(K_SAMPLES))
    conf = float(rng.beta(9, 1.5)) if "confident" in pop else float(rng.beta(4, 3))
    claim = G.Claim(f"claim-{cid}", "regulatory_fact", stakes, _sources(rng, pop, cid), conf, samples)
    wrong = pop.startswith("wrong")
    delay = float(np.clip(math.exp(rng.normal(math.log(4.0), 1.0)), 0.25, 90.0))
    nd = int(rng.integers(1, 6))
    return Item(claim, pop, wrong, 0.0, delay, nd, [bool(rng.random() < .3) for _ in range(nd)])


def generate(seed: int, n: int, span: float = ARRIVAL_SPAN_DAYS) -> list[Item]:
    rng = np.random.default_rng(seed)
    items = [gen_claim(rng, i) for i in range(n)]
    ts = np.sort(rng.uniform(0, span, n))
    for it, t in zip(items, ts):
        it.t = float(t)
    return items


def calibrate(seed: int = SEED + 1) -> dict:
    """Holdout draw, separate from the evaluation stream. Returns tier taus and arm B's tau."""
    calib = generate(seed, N_CALIB)
    wrong = [G.agreement(i.claim.samples) for i in calib if i.wrong]
    return {"tiers": {n: G.conformal_threshold(wrong, t.alpha) for n, t in G.TIERS.items()},
            "b": G.conformal_threshold(wrong, ARM_B_ALPHA), "n_wrong": len(wrong)}


# ------------------------------------------------------------------ arms
def decide_arm(arm: str, item: Item, cal: dict) -> G.Decision:
    if arm == "A":
        return G.Decision(G.EMIT)
    if arm == "B":
        return G.decide_conformal_only(item.claim, cal["b"])
    return G.decide(item.claim, cal["tiers"])


def run_arm(arm: str, items: list[Item], cal: dict) -> dict:
    contain_holds_cancel = arm in ("C", "D")
    ts = Timestream()
    shard = lambda i: f"s{i % N_SHARDS}"
    for s in range(N_SHARDS):
        ts.branches[shard(s)] = None
    m = dict(arm=arm, n=len(items), emit=0, contain=0, unknown=0, wrong_emitted=0, wrong_contained=0,
             wrong_caught=0, wc_total=0, wc_caught=0, wc_contained=0, correct_unknown=0,
             contaminated_internal=0, contaminated_external=0, ext_cancelled=0, ach_overrides=0,
             repaired_internal=0, repaired_external=0, comp_records=0, ttr=[])
    # build the time-ordered action list
    actions = []   # (t, order, kind, idx, extra)
    for idx, it in enumerate(items):
        d = decide_arm(arm, it, cal)
        if d.reasons and d.reasons[0] == "ach":
            m["ach_overrides"] += 1
        it._d = d
        if it.pop == "wrong_confident":
            m["wc_total"] += 1
        if d.outcome == G.UNKNOWN:
            m["unknown"] += 1
            if it.wrong:
                m["wrong_caught"] += 1
            if it.pop == "wrong_confident":
                m["wc_caught"] += 1
            if not it.wrong:
                m["correct_unknown"] += 1
            continue
        m["emit" if d.outcome == G.EMIT else "contain"] += 1
        if it.wrong:
            m["wrong_emitted"] += 1
            if d.outcome == G.CONTAIN:
                m["wrong_contained"] += 1
                if it.pop == "wrong_confident":
                    m["wc_contained"] += 1
        actions.append((it.t, 0, "claim", idx))
    for a in list(actions):
        idx = a[3]
        it = items[idx]
        contain = it._d.outcome == G.CONTAIN
        for j in range(it.n_decisions):
            if it.ext_flags[j] and contain:
                actions.append((it.t + 1e-6 * (j + 1), 1, "hold", idx, j))
                release_t = it.t + HOLD_DAYS
                refuted_before = it.wrong and contain_holds_cancel and it.delay < HOLD_DAYS
                if refuted_before:
                    actions.append((it.t + it.delay, 2, "cancel", idx, j))
                    if it.wrong:
                        m["ext_cancelled"] += 1
                else:
                    actions.append((release_t, 1, "release", idx, j))
            elif it.ext_flags[j]:
                actions.append((it.t + 1e-6 * (j + 1), 1, "ext", idx, j))
            else:
                actions.append((it.t + 1e-6 * (j + 1), 1, "decision", idx, j))
    actions.sort(key=lambda a: (a[0], a[1], a[3], a[4] if len(a) > 4 else -1))
    claim_event = {}
    for a in actions:
        t, _, kind, idx = a[:4]
        it = items[idx]
        br = shard(idx)
        if kind == "claim":
            ev = ts.append(br, {"kind": "claim", "key": f"c{idx}", "t": round(t, 6),
                                "outcome": it._d.outcome}, INTERNAL)
            claim_event[idx] = ev.id
            continue
        j = a[4]
        base = {"cause": f"c{idx}", "t": round(t, 6), "n": j}
        if kind == "decision":
            ts.append(br, dict(base, kind="decision", key=f"c{idx}:d{j}"), INTERNAL)
            if it.wrong: m["contaminated_internal"] += 1
        elif kind == "ext":
            ts.append(br, dict(base, kind="ext_decision", key=f"c{idx}:d{j}"), EXTERNAL)
            if it.wrong: m["contaminated_external"] += 1
        elif kind == "hold":
            ts.append(br, dict(base, kind="hold", key=f"c{idx}:h{j}"), INTERNAL)
        elif kind == "release":
            ts.append(br, dict(base, kind="release", key=f"c{idx}:d{j}"), EXTERNAL)
            if it.wrong: m["contaminated_external"] += 1
        elif kind == "cancel":
            ts.append(br, dict(base, kind="cancel", key=f"c{idx}:x{j}"), INTERNAL)
    m["main_verified"] = all(ts.verify(shard(s)) for s in range(N_SHARDS))
    m["head_digest"] = _digest(ts, [shard(s) for s in range(N_SHARDS)])
    if arm == "D":
        _repair(ts, items, claim_event, shard, m)
        m["fix_digest"] = _digest(ts, sorted(b for b in ts.branches if b.startswith("fix-")))
    m["mean_ttr"] = float(np.mean(m["ttr"])) if m["ttr"] else None
    return m


def _digest(ts, branches):
    import hashlib
    h = hashlib.sha256()
    for b in branches:
        h.update((ts.branches[b] or "").encode())
    return h.hexdigest()[:16]


def _repair(ts, items, claim_event, shard, m):
    """On detection, fork the shard at the claim event and replay with the claim retracted."""
    detections = sorted((items[i].t + items[i].delay, i) for i in claim_event if items[i].wrong)
    for t_detect, idx in detections:
        br, bad = shard(idx), claim_event[idx]
        fix = f"fix-{idx}"
        ts.fork(br, bad, fix)
        suffix = ts.descendants(br, bad)
        key = f"c{idx}"

        def transform(ev, key=key):
            p = {k: v for k, v in ev.payload.items() if k != "effect_class"}
            if p.get("key") == key or p.get("cause") == key:
                return None          # retract the claim and everything derived from it
            return p

        rep = ts.replay(fix, suffix, transform)
        m["repaired_internal"] += sum(1 for e in rep.repaired if e.kind == "decision")
        m["repaired_external"] += sum(1 for e in rep.repaired if e.effect_class == EXTERNAL)
        m["comp_records"] += len(rep.compensation_required)
        m["ttr"].append(t_detect - items[idx].t)


COLS = [
    ("wrong claims emitted (EMIT+CONTAIN)", "wrong_emitted"),
    ("  of which CONTAINed", "wrong_contained"),
    ("wrong claims caught (UNKNOWN)", "wrong_caught"),
    ("wrong-confident total", "wc_total"),
    ("wrong-confident caught (UNKNOWN)", "wc_caught"),
    ("wrong-confident CONTAINed", "wc_contained"),
    ("abstention rate", "abst"),
    ("correct claims abstained on", "correct_unknown"),
    ("external holds cancelled (wrong claims)", "ext_cancelled"),
    ("contaminated decisions (internal)", "contaminated_internal"),
    ("contaminated decisions (external, executed)", "contaminated_external"),
    ("contaminated repaired by fork (internal)", "repaired_internal"),
    ("external effects repaired by fork", "repaired_external"),
    ("external effects needing compensation", "comp"),
    ("mean time-to-repair (days)", "ttr"),
]


def table(results: list[dict]) -> str:
    rows = []
    for label, k in COLS:
        cells = []
        for r in results:
            if k == "abst":
                v = f"{r['unknown'] / r['n']:.1%}"
            elif k == "comp":
                # arms without replay never produce compensation records; the executed external
                # effects of wrong claims exist anyway and nobody is told about them.
                v = f"{r['comp_records']}" if r["arm"] == "D" else f"{r['contaminated_external']} (unflagged)"
            elif k == "ttr":
                v = f"{r['mean_ttr']:.2f}" if r["mean_ttr"] is not None else "n/a"
            else:
                v = str(r[k])
            cells.append(v)
        rows.append((label, cells))
    w = max(len(l) for l, _ in rows)
    head = f"{'':<{w}} | " + " | ".join(f"{'Arm ' + r['arm']:>14}" for r in results)
    out = [head, "-" * len(head)]
    for l, c in rows:
        out.append(f"{l:<{w}} | " + " | ".join(f"{x:>14}" for x in c))
    return "\n".join(out)


def conformal_check(cal: dict, seed: int = SEED + 2) -> list[tuple[str, float, float, float]]:
    """Fresh draw: share of WRONG claims whose agreement exceeds each tier threshold."""
    fresh = [i for i in generate(seed, N_CALIB) if i.wrong]
    out = []
    for name, tier in G.TIERS.items():
        tau = cal["tiers"][name]
        passed = sum(1 for i in fresh if G.agreement(i.claim.samples) > tau) / len(fresh)
        out.append((name, tier.alpha, tau, passed))
    return out


def main() -> int:
    cal = calibrate()
    items = generate(SEED, N_CLAIMS)
    results = [run_arm(a, items, cal) for a in "ABCD"]
    print(f"forkgate sim  seed={SEED}  N={N_CLAIMS}  calib_wrong={cal['n_wrong']}  hold={HOLD_DAYS}d")
    pops = {p: sum(1 for i in items if i.pop == p) for p in POPS}
    print("populations:", ", ".join(f"{p}={c}" for p, c in pops.items()))
    print("conformal thresholds (agreement > tau to pass): "
          + ", ".join(f"{n}={t:.3f}" for n, t in cal["tiers"].items()) + f", armB={cal['b']:.3f}")
    print()
    print(table(results))
    print()
    print("Out-of-sample conformal check (fresh draw, P(emit | wrong) vs alpha):")
    for name, alpha, tau, passed in conformal_check(cal):
        print(f"  {name:<6} alpha={alpha:.2f} tau={tau:.3f} observed={passed:.3f}")
    A, B, C, D = results
    wc = B["wc_total"]
    print()
    print("Pre-registered predictions:")
    p1 = B["wc_caught"] / wc
    print(f"  1 (B) wrong-confident caught = {B['wc_caught']}/{wc} = {p1:.1%}  (need < 20%)  -> {'PASS' if p1 < .20 else 'FAILED'}")
    c_flag = (C["wc_caught"] + C["wc_contained"]) / wc
    ok2 = c_flag >= .40 and p1 > 0 and c_flag >= 3 * p1
    print(f"  2 (C) wrong-confident flagged = {C['wc_caught'] + C['wc_contained']}/{wc} = {c_flag:.1%} vs B {p1:.1%}"
          f"  (need >=40% and >=3x B)  -> {'PASS' if ok2 else 'FAILED'}")
    ok3 = (D["repaired_internal"] == D["contaminated_internal"] and D["repaired_external"] == 0
           and D["comp_records"] == D["contaminated_external"] and D["comp_records"] > 0)
    print(f"  3 (D) internal repaired {D['repaired_internal']}/{D['contaminated_internal']}, external repaired "
          f"{D['repaired_external']}, compensation_required {D['comp_records']} vs executed external "
          f"{D['contaminated_external']}  -> {'PASS' if ok3 else 'FAILED'}")
    ratio = D["comp_records"] / A["contaminated_external"]
    print(f"  4 (D) compensation {D['comp_records']} vs A {A['contaminated_external']} = {ratio:.2f}x (need < 0.60)"
          f"  -> {'PASS' if ratio < .6 else 'FAILED'}")
    ok5 = all(p <= a + .02 for _, a, _, p in conformal_check(cal))
    print(f"  5 conformal out-of-sample pass-through <= alpha+0.02 at every tier  -> {'PASS' if ok5 else 'FAILED'}")
    print()
    print(f"ACH overrides (arms C/D): {C['ach_overrides']}   main-chain verified: {all(r['main_verified'] for r in results)}")
    print(f"branch digests  main(D)={D['head_digest']}  fix(D)={D['fix_digest']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
