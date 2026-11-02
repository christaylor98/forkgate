"""A minimal model of axFlow-style fork semantics, not axFlow itself.

Append-only, content-addressed, parent-linked event log with named branch heads.
Fork at a bad event, replay the downstream with a correction, diff the branches.

The load-bearing limit: events carry an effect_class. "internal" events live in the
stream and are fully forkable. "external" events (email sent, payment made, API call)
have already left the system. Replay can never repair them; it emits a
compensation_required record instead.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

INTERNAL = "internal"
EXTERNAL = "external"
GENESIS = "0" * 64


def canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def event_hash(parent: Optional[str], payload: dict) -> str:
    h = hashlib.sha256()
    h.update((parent or GENESIS).encode())
    h.update(canonical(payload).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class Event:
    id: str
    parent: Optional[str]
    effect_class: str
    payload: dict

    @property
    def key(self) -> str:
        return self.payload["key"]

    @property
    def kind(self) -> str:
        return self.payload.get("kind", "")


@dataclass
class ReplayReport:
    new_branch: str
    unchanged: list = field(default_factory=list)
    repaired: list = field(default_factory=list)  # internal events corrected or dropped
    compensation_required: list = field(default_factory=list)  # external, cannot be undone

    def summary(self) -> str:
        return (f"replay -> {self.new_branch}: {len(self.unchanged)} unchanged, "
                f"{len(self.repaired)} repaired (internal), "
                f"{len(self.compensation_required)} compensation_required (external)")


class Timestream:
    def __init__(self) -> None:
        self.events: dict[str, Event] = {}
        self.branches: dict[str, Optional[str]] = {"main": None}

    # -- writing ---------------------------------------------------------
    def append(self, branch: str, payload: dict, effect_class: str = INTERNAL) -> Event:
        if effect_class not in (INTERNAL, EXTERNAL):
            raise ValueError(f"bad effect_class {effect_class!r}")
        if "key" not in payload:
            raise ValueError("payload needs a logical 'key'")
        parent = self.branches[branch]
        body = dict(payload, effect_class=effect_class)
        ev = Event(event_hash(parent, body), parent, effect_class, body)
        self.events[ev.id] = ev
        self.branches[branch] = ev.id
        return ev

    # -- reading ---------------------------------------------------------
    def history(self, branch: str) -> list[Event]:
        out, cur = [], self.branches[branch]
        while cur is not None:
            ev = self.events[cur]
            out.append(ev)
            cur = ev.parent
        out.reverse()
        return out

    def verify(self, branch: str) -> bool:
        """Recompute every hash on the branch and check the parent links."""
        prev = None
        for ev in self.history(branch):
            if ev.parent != prev or event_hash(ev.parent, ev.payload) != ev.id:
                return False
            prev = ev.id
        return True

    # -- forking ---------------------------------------------------------
    def fork(self, branch: str, at_event_id: str, new_branch: str) -> str:
        """New branch whose history is the prefix of `branch` up to, EXCLUDING, at_event_id."""
        if new_branch in self.branches:
            raise ValueError(f"branch {new_branch!r} exists")
        ids = {e.id for e in self.history(branch)}
        if at_event_id not in ids:
            raise KeyError("event not on branch")
        self.branches[new_branch] = self.events[at_event_id].parent
        return new_branch

    def descendants(self, branch: str, at_event_id: str) -> list[Event]:
        """Events on `branch` from at_event_id (inclusive) to head, oldest first."""
        h = self.history(branch)
        idx = next(i for i, e in enumerate(h) if e.id == at_event_id)
        return h[idx:]

    def replay(self, new_branch: str, events: Iterable[Event],
               transform: Callable[[Event], Optional[dict]]) -> ReplayReport:
        """Re-run `events` on new_branch. transform(event) returns the corrected payload,
        the same payload (unaffected), or None (the event should not have happened).

        Internal events are rewritten in the new branch. External events that are
        affected cannot be undone: a compensation_required record is appended instead,
        and the event is never reported as repaired."""
        rep = ReplayReport(new_branch)
        for ev in events:
            orig = {k: v for k, v in ev.payload.items() if k != "effect_class"}
            new = transform(ev)
            if new is not None and new == orig:
                self.append(new_branch, new, ev.effect_class)
                rep.unchanged.append(ev)
            elif ev.effect_class == EXTERNAL:
                rec = {"kind": "compensation_required", "key": f"comp:{ev.key}",
                       "original_event": ev.id, "original_key": ev.key,
                       "original": orig, "corrected": new,
                       "reason": "external effect already left the system; fork cannot undo it"}
                self.append(new_branch, rec, INTERNAL)
                rep.compensation_required.append(rec)
            else:
                if new is not None:
                    self.append(new_branch, new, INTERNAL)
                rep.repaired.append(ev)
        return rep

    # -- comparing -------------------------------------------------------
    def diff(self, a: str, b: str) -> dict:
        """Which downstream decisions changed between two branches, matched on logical key."""
        ka = {e.key: e for e in self.history(a)}
        kb = {e.key: e for e in self.history(b)}
        removed = sorted(k for k in ka if k not in kb)
        added = sorted(k for k in kb if k not in ka)
        changed = sorted(k for k in ka if k in kb and ka[k].payload != kb[k].payload)
        same = sum(1 for k in ka if k in kb and ka[k].payload == kb[k].payload)
        return {"removed": removed, "added": added, "changed": changed, "unchanged": same}
