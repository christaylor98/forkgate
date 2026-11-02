"""One case, start to finish. Run: python scenario.py"""
from forkgate import gates as G
from forkgate.sim import calibrate
from forkgate.timestream import EXTERNAL, INTERNAL, Timestream

HOLD_DAYS, REFUTED_DAY = 2, 4


def rule(title):
    print(f"\n=== {title} " + "=" * (66 - len(title)))


claim = G.Claim(
    text="Regulation threshold for storing compound X is 50 units",
    claim_type="regulatory_fact", stakes="medium",
    sources=tuple(G.Source("industry-blog-2019", (40, 45)) for _ in range(3)),  # 3 pages, ONE origin
    model_confidence=0.97, samples=("50 units",) * 19 + ("500 units",))

rule("1. the claim and the gate")
cal = calibrate()
d = G.decide(claim, cal["tiers"])
rel, cred = G.admiralty(claim)
print(f'claim     : "{claim.text}"')
print(f"stakes    : {claim.stakes} ({G.TIERS['medium'].standard}), alpha={G.TIERS['medium'].alpha}")
print(f"agreement : {G.agreement(claim.samples):.2f} vs conformal tau={cal['tiers']['medium']:.2f}  -> passes (confident)")
print(f"citations : {len(claim.sources)} listed, {G.origin_counts(claim.sources)[0]} distinct origin")
print(f"admiralty : reliability {rel} / credibility {cred}  (medium needs "
      f"{G.TIERS['medium'].min_reliability} / <= {G.TIERS['medium'].max_credibility})")
print(f"ACH       : least-refuted hypothesis = {G.ach(claim)!r}")
print(f"OUTCOME   : {d.outcome}   reasons={d.reasons}")
assert d.outcome == G.CONTAIN

ts = Timestream()
ts.append("main", {"kind": "note", "key": "n0", "t": 0.0, "text": "unrelated earlier event"})
c = ts.append("main", {"kind": "claim", "key": "claim-1", "t": 1.0, "outcome": d.outcome})
ts.append("main", {"kind": "decision", "key": "d1", "cause": "claim-1", "t": 1.1,
                   "what": "update compliance checklist (limit=50)"})
ts.append("main", {"kind": "decision", "key": "d2", "cause": "claim-1", "t": 1.2,
                   "what": "set storage cap in capacity plan (50)"})
ts.append("main", {"kind": "hold", "key": "h3", "cause": "claim-1", "t": 1.3,
                   "what": "HELD: email supplier 'limit is 50'"})
ts.append("main", {"kind": "note", "key": "n1", "t": 2.0, "text": "unrelated later event"})
ts.append("main", {"kind": "release", "key": "d3", "cause": "claim-1", "t": 1.0 + HOLD_DAYS,
                   "what": "EMAIL SENT to supplier: 'limit is 50'"}, EXTERNAL)

rule("2. timestream on main (CONTAIN: externals held, then released after the hold window)")
for e in ts.history("main"):
    mark = "EXTERNAL" if e.effect_class == EXTERNAL else "internal"
    print(f"  day {e.payload['t']:>4}  {e.id[:8]}  [{mark}]  {e.payload.get('what', e.payload.get('text', e.kind))}")
print(f"  hash chain verified: {ts.verify('main')}")

rule(f"3. day {REFUTED_DAY}: delayed oracle refutes the claim (true limit: 5 units)")
print("  claim-1 REFUTED. The email left the system on day 3. Fork at the claim.")

ts.fork("main", c.id, "fix")
rep = ts.replay("fix", ts.descendants("main", c.id),
                lambda e: None if (e.key == "claim-1" or e.payload.get("cause") == "claim-1")
                else {k: v for k, v in e.payload.items() if k != "effect_class"})

rule("4. fork at claim-1, replay downstream with the claim retracted")
print(f"  fix branch starts at {ts.history('fix')[0].id[:8]} (prefix EXCLUDES the bad event)")
print("  " + rep.summary())

rule("5. diff main vs fix")
df = ts.diff("main", "fix")
byk = {e.key: e for e in ts.history("main")}
for k in df["removed"]:
    e = byk[k]
    if e.kind in ("decision",):
        print(f"  REPAIRED   {k:<8} [internal] {e.payload['what']}")
    elif e.kind in ("hold", "claim"):
        print(f"  (retracted) {k:<8} [internal] {e.kind}")
for k in df["added"]:
    r = [x for x in rep.compensation_required if x["key"] == k][0]
    print(f"  COMPENSATE {r['original_key']:<8} [EXTERNAL] {r['original']['what']}")
    print(f"             -> {r['reason']}")
print(f"  unchanged events carried over: {df['unchanged']}")

repaired_decisions = [e for e in rep.repaired if e.kind == "decision"]
rule("result")
print(f"  decisions repaired by fork : {len(repaired_decisions)}  (internal)")
print(f"  flagged for compensation   : {len(rep.compensation_required)}  (external, NOT undone)")
print("""
       claim --CONTAIN--> [d1 internal] [d2 internal] [d3 EXTERNAL email]
                               |            |              |
                fork+replay:  repaired    repaired    compensation_required
""")
assert len(repaired_decisions) == 2 and len(rep.compensation_required) == 1
