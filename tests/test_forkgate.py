import math

import numpy as np
import pytest

from forkgate import gates as G
from forkgate import sim
from forkgate.timestream import EXTERNAL, INTERNAL, Timestream


def build(n=6):
    ts = Timestream()
    evs = [ts.append("main", {"kind": "note", "key": f"e{i}", "i": i}) for i in range(n)]
    return ts, evs


# ---- timestream
def test_hash_chain_integrity():
    ts, evs = build()
    assert ts.verify("main")
    assert all(evs[i].parent == evs[i - 1].id for i in range(1, len(evs)))
    # tamper with a stored payload: verification must fail
    evs[2].payload["i"] = 999
    assert not ts.verify("main")


def test_fork_excludes_bad_event():
    ts, evs = build()
    ts.fork("main", evs[3].id, "fix")
    ids = [e.id for e in ts.history("fix")]
    assert ids == [e.id for e in evs[:3]]
    assert evs[3].id not in ids


def _run_replay(seed):
    rng = np.random.default_rng(seed)
    ts = Timestream()
    bad = None
    for i in range(20):
        kind = "claim" if i == 5 else "decision"
        ev = ts.append("main", {"kind": kind, "key": f"e{i}", "cause": "e5" if i > 5 and rng.random() < .5 else None},
                       EXTERNAL if (i > 5 and rng.random() < .3) else INTERNAL)
        bad = ev if i == 5 else bad
    ts.fork("main", bad.id, "fix")
    ts.replay("fix", ts.descendants("main", bad.id),
              lambda e: None if e.payload.get("cause") == "e5" or e.key == "e5"
              else {k: v for k, v in e.payload.items() if k != "effect_class"})
    return [e.id for e in ts.history("fix")]


def test_replay_determinism():
    assert _run_replay(7) == _run_replay(7)
    assert _run_replay(7) != _run_replay(8)


def test_external_effects_never_marked_repaired():
    ts = Timestream()
    ts.append("main", {"kind": "note", "key": "n"})
    c = ts.append("main", {"kind": "claim", "key": "c"})
    ts.append("main", {"kind": "decision", "key": "d1", "cause": "c"})
    ts.append("main", {"kind": "release", "key": "d2", "cause": "c"}, EXTERNAL)
    ts.fork("main", c.id, "fix")
    rep = ts.replay("fix", ts.descendants("main", c.id),
                    lambda e: None if e.key == "c" or e.payload.get("cause") == "c" else dict(e.payload))
    assert all(e.effect_class == INTERNAL for e in rep.repaired)
    assert [r["original_key"] for r in rep.compensation_required] == ["d2"]
    assert rep.compensation_required[0]["kind"] == "compensation_required"
    d = ts.diff("main", "fix")
    assert "d2" in d["removed"] and "comp:d2" in d["added"]


def test_sim_never_repairs_external():
    items = sim.generate(1, 600)
    r = sim.run_arm("D", items, sim.calibrate())
    assert r["repaired_external"] == 0
    assert r["comp_records"] == r["contaminated_external"] > 0
    assert r["repaired_internal"] == r["contaminated_internal"]


# ---- admiralty
def S(origin, rec=(40, 45), supports=True):
    return G.Source(origin, rec, supports)


def test_admiralty_axes_independent():
    # vary corroboration only: reliability unchanged
    base = [S("a")]
    more = [S("a"), S("b", (40, 45)), S("c", (40, 45))]
    assert G.reliability_grade(G.reliability_inputs(base)) == G.reliability_grade(G.reliability_inputs(more))
    assert G.credibility_grade(*G.origin_counts(base)) != G.credibility_grade(*G.origin_counts(more))
    # vary track record only: credibility unchanged
    good, bad = [S("a", (49, 50))], [S("a", (10, 50))]
    assert G.credibility_grade(*G.origin_counts(good)) == G.credibility_grade(*G.origin_counts(bad))
    assert G.reliability_grade(G.reliability_inputs(good)) != G.reliability_grade(G.reliability_inputs(bad))


def test_shared_origin_counts_once():
    echoes = [S("same") for _ in range(5)]
    assert G.origin_counts(echoes) == (1, 0)
    assert G.credibility_grade(*G.origin_counts(echoes)) == G.credibility_grade(*G.origin_counts([S("same")]))
    assert G.credibility_grade(*G.origin_counts([S("x"), S("y")])) < G.credibility_grade(*G.origin_counts(echoes))


def test_F_is_not_E():
    assert G.reliability_grade([None]) == "F"
    assert G.reliability_grade([(1, 2)]) == "F"          # too little history
    assert G.reliability_grade([(0, 50)]) == "E"         # judged and bad
    assert G.credibility_grade(0, 0) == 6
    tier = G.TIERS["low"]
    assert not G.admiralty_ok("F", 1, tier)


def test_contradiction_gives_5():
    assert G.credibility_grade(1, 2) == 5


# ---- conformal
def test_conformal_quantile_index():
    scores = list(range(1, 100))          # n = 99
    # alpha=.1 -> k = ceil(100*.9) = 90 -> 90th smallest
    assert G.conformal_threshold(scores, 0.1) == 90
    assert G.conformal_threshold([1, 2, 3], 0.01) == math.inf   # k = 4 > n


def test_conformal_risk_below_alpha_on_heldout():
    cal = sim.calibrate()
    for name, alpha, tau, observed in sim.conformal_check(cal):
        assert observed <= alpha + 0.02, (name, alpha, observed)


# ---- ACH
def test_ach_steps_are_separate_and_least_refuted_wins():
    claim = G.Claim("t", "x", "low", (S("a"), S("b", supports=False), S("c", supports=False)), .9,
                    ("A",) * 11 + ("B",) * 9)
    hyps = G.ach_generate_hypotheses(claim)
    assert hyps == ["A", "B"]
    scores = {h: G.ach_evaluate(claim, h) for h in hyps}
    assert G.ach_select(scores) == min(scores, key=scores.get)
