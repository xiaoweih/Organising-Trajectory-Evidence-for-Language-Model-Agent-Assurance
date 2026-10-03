"""The assurance model over tau2-bench telecom traces (Section 3 of the paper).

A `Run` wraps a canonical `adapter.Trace`.  Positions are the canonical steps.
The atomic propositions of Definition 3.10 are provided as `Atom`s built from
the log prefix; the argument structure D_{tau,t} = ext_D(s^a_t) is built
deterministically (the pilot's two stand-ins G1/G2 as argument structures).
Everything here is programmatic: no model call.
"""
from __future__ import annotations
import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapter import Trace, Step
import constraints as K
import grounding as GR
from .logic import (Atom, ArgStructure, Forall, Exists)


# ---------------------------------------------------------------- action instances
class ActionInstance:
    """A tool call with its arguments, or a message."""
    __slots__ = ("kind", "tool", "args", "idx")
    def __init__(self, kind, tool=None, args=None, idx=None):
        self.kind, self.tool, self.args, self.idx = kind, tool, args or {}, idx
    def __repr__(self):
        return f"{self.tool}({self.args})" if self.kind == "tool_call" else f"msg@{self.idx}"
    def __hash__(self): return hash((self.kind, self.tool, self.idx))
    def __eq__(self, o): return isinstance(o, ActionInstance) and (self.kind, self.tool, self.idx) == (o.kind, o.tool, o.idx)


class Run:
    def __init__(self, trace: Trace, guard=None):
        self.t = trace
        self.steps = trace.steps
        self.guard = guard                     # g(local_state_view, alpha) -> bool, or None
        self._lines_cache = {}
        self._arg_cache = {}
        # blocked positions under counterfactual gating: the guard is applied to
        # the proposed call; the rest of the (recorded) run is unchanged.
        self.blocked = set()
        if guard is not None:
            for s in self.steps:
                if s.kind == "tool_call" and s.actor == "assistant":
                    if guard(self, s.idx, ActionInstance("tool_call", s.tool, s.args, s.idx)):
                        self.blocked.add(s.idx)

    def positions(self): return len(self.steps)

    # ---- what occurs at a position -------------------------------------------
    def occurring(self, t):
        s = self.steps[t]
        if s.kind == "tool_call" and s.actor == "assistant":
            return [ActionInstance("tool_call", s.tool, s.args, s.idx)]
        if s.kind == "message" and s.actor == "assistant":
            return [ActionInstance("message", idx=s.idx)]
        return []

    # ---- event propositions ----------------------------------------------------
    def proposed(self, t, tool=None, pred=None):
        s = self.steps[t]
        if not (s.kind == "tool_call" and s.actor == "assistant"): return False
        if tool is not None and s.tool != tool: return False
        return pred(s) if pred else True

    def executed(self, t, tool=None, pred=None):
        return self.proposed(t, tool, pred) and t not in self.blocked

    def is_blocked(self, t): return t in self.blocked

    # ---- domain propositions (most recent value in the log at or before t) --------
    def line_records(self, t):
        """Line records the assistant has observed up to position t (last one wins)."""
        if t in self._lines_cache: return self._lines_cache[t]
        out = {}
        for s in self.steps[:t + 1]:
            if s.kind != "tool_result" or s.error or s.tool != "get_details_by_id": continue
            try: p = json.loads(s.text)
            except Exception: continue
            if isinstance(p, dict) and "line_id" in p:
                out[str(p["line_id"])] = p
        self._lines_cache[t] = out
        return out

    def expired_lines(self, t):
        return {l for l, p in self.line_records(t).items()
                if str(p.get("contract_end_date") or "9999") < K.DOMAIN_NOW}

    def roaming_off_lines(self, t):
        return {l for l, p in self.line_records(t).items() if p.get("roaming_enabled") is False}

    def user_said_abroad(self, t):
        return any(K.ABROAD_RE.search(s.text) for s in self.steps[:t + 1]
                   if s.kind == "message" and s.actor == "user")

    def called_before(self, t, tools, actor="assistant", strict=True):
        rng = self.steps[:t] if strict else self.steps[:t + 1]
        return any(s.kind == "tool_call" and s.tool in tools and (actor is None or s.actor == actor) for s in rng)

    def details_call(self, t, prefix, actor="assistant"):
        s = self.steps[t]
        return (s.kind == "tool_call" and (actor is None or s.actor == actor)
                and s.tool == "get_details_by_id" and str(s.args.get("id", "")).upper().startswith(prefix))

    # ---- task metadata (assessor-side, not in the assistant's log) -----------------
    @property
    def issue_type(self): return self.t.issue_type
    @property
    def faults(self): return self.t.faults

    # ---- argument structure (deterministic ext_D) ------------------------------
    def argstruct(self, t):
        """D_{tau,t} for the two pilot stand-ins.
        Nodes: for a refuel call at t, arg(refuel) rests on premise 'exhausted(l)';
        for a closing 'fixed' claim at t, arg(claim) rests on premise 'healthy-after-fix'.
        A premise stands iff the assistant's own observations support it."""
        if t in self._arg_cache: return self._arg_cache[t]
        D = ArgStructure()
        s = self.steps[t]
        if s.kind == "tool_call" and s.actor == "assistant" and s.tool == "refuel_data":
            a = ActionInstance("tool_call", s.tool, s.args, s.idx)
            line = s.args.get("line_id")
            used = limit = None
            for r in self.steps[:t]:
                if r.kind != "tool_result" or r.error or r.tool not in {"get_data_usage", "get_details_by_id"}: continue
                try: p = json.loads(r.text)
                except Exception: continue
                if not isinstance(p, dict) or p.get("line_id") not in (None, line): continue
                if p.get("data_used_gb") is not None: used = float(p["data_used_gb"])
                if p.get("data_limit_gb") is not None: limit = float(p["data_limit_gb"])
            D.V |= {"refuel", "exhausted"}
            D.dep["refuel"] = {"exhausted"}
            D.St.add("refuel")
            if used is not None and limit is not None and used >= limit:
                D.St.add("exhausted")
            D.arg[a] = "refuel"
        elif s.kind == "message" and s.actor == "assistant" and self._is_closing_claim(t):
            a = ActionInstance("message", idx=s.idx)
            fixes = [x.idx for x in self.t.calls() if x.tool in GR.STATE_CHANGING and x.idx < t]
            last_fix = max(fixes) if fixes else -1
            window = [x for x in self.steps if last_fix < x.idx < t]
            supported = any(GR._confirms(x) is True for x in window)
            D.V |= {"claim", "healthy"}
            D.dep["claim"] = {"healthy"}
            D.St.add("claim")
            if supported: D.St.add("healthy")
            D.arg[a] = "claim"
        self._arg_cache[t] = D
        return D

    def _is_closing_claim(self, t):
        """Prefix-only candidate rule: an assistant message at t is a closing-claim
        candidate iff its own text matches the claim pattern.  (The pilot's original
        predicate looked at the last three messages of the *completed* run, which is
        retrospective and not a function of the decision prefix; see the paper, E7.)"""
        s = self.steps[t]
        return (s.kind == "message" and s.actor == "assistant"
                and bool(GR.CLAIM_RE.search(s.text)) and not GR.NOT_CLAIM_RE.search(s.text))

    def checked(self, t):
        """checked(alpha): alpha was presented to the support checker at t."""
        s = self.steps[t]
        if s.kind == "tool_call" and s.actor == "assistant" and s.tool == "refuel_data": return True
        if s.kind == "message" and s.actor == "assistant" and self._is_closing_claim(t): return True
        return False


# ---------------------------------------------------------------- atom builders
def proposed(tool, pred=None, name=None):
    return Atom(name or f"proposed({tool})", lambda r, t: r.proposed(t, tool, pred))

def executed(tool, pred=None, name=None):
    return Atom(name or f"executed({tool})", lambda r, t: r.executed(t, tool, pred))

def proposed_any(tools, name):
    return Atom(name, lambda r, t: r.proposed(t) and r.steps[t].tool in tools)

def called_any(tools, name, actor=None):
    """proposed at t by `actor` (None = either actor): device-side actions are the user's."""
    def fn(r, t):
        s = r.steps[t]
        return s.kind == "tool_call" and s.tool in tools and (actor is None or s.actor == actor)
    return Atom(name, fn, local=True)   # user-side device actions reach the assistant as observations

def details(prefix, name, actor="assistant"):
    return Atom(name, lambda r, t: r.details_call(t, prefix, actor))

def assistant_msg(pred=None, name="assistantMsg"):
    def fn(r, t):
        s = r.steps[t]
        return s.kind == "message" and s.actor == "assistant" and (pred(s) if pred else True)
    return Atom(name, fn)

def domain(name, fn, local=True):
    return Atom(name, fn, local=local)
