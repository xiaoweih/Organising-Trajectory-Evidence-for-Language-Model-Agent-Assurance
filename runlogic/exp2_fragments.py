"""E2: fragment membership of the twenty rules, decided syntactically (Section 3.5),
and the expressiveness gap of Pol_C under growing method sets (Section 7)."""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runlogic.logic import is_forward, is_local, uses_argument_formula, G, Not, U, TRUE
from runlogic.rules_telecom import RULES
import json as _j

# rules whose forward-looking obligation is a prohibition of an identifiable action
# instance admit the step-wise reformulation G((O rho) -> ~proposed(a)) (Section 7.3)
REFORMULABLE = {"C07"}

# L_act rules whose guard would have to read more of the local state than the
# pilot's guard class does (the proposed call's head and the line records):
# C10 (one call per message) needs the preceding position of the same message,
# C11 (no prose with a call) needs the message text at the previous position,
# C13 (network check before SIM) needs the customer's device actions, which reach
# the assistant as observations.  All three are in L_act (a guard reads s^a_t);
# none is decidable by a call-head guard.  Implementation classification (E2).
NOT_CALL_HEAD = {"C10", "C11", "C13"}

def classify(rule):
    fwd = is_forward(rule.chi)
    loc = is_local(rule.rho) and is_local(rule.chi)
    oracle_app = not is_local(rule.rho)
    stepwise = (not fwd) and loc
    if rule.cid in REFORMULABLE and loc: stepwise = True
    single = stepwise and rule.cid not in NOT_CALL_HEAD
    return {"cid": rule.cid, "text": rule.text, "forward": fwd, "local": loc,
            "oracle_applicability": oracle_app, "stepwise": stepwise,
            "call_head_gateable": single,
            "fragment": ("L_act (gateable)" if stepwise else
                         "L_rule, assessor only (forward-looking)" if fwd and not oracle_app else
                         "L_rule, task-metadata applicability" if oracle_app else
                         "L_rule, assessor only")}

def main(exp1_path, out):
    e1 = json.load(open(exp1_path))["summary"]["per_rule"]
    rows = [classify(r) for r in RULES]
    for r in rows:
        r["app"] = e1[r["cid"]]["app"]; r["viol"] = e1[r["cid"]]["viol"]
    # expressiveness gap of Pol_C under method sets
    sets = {
        "{C5 reach}": lambda r: r["stepwise"],            # any guard over s^a_t
        "{C5 configured}": lambda r: r["cid"] == "C07",   # the pilot's one guard
        "{C1}": lambda r: True,                       # every extracted rule is addressed by C1 (offline)
        "{C1,C5}": lambda r: True,
        "{C4}": lambda r: False,                      # groundedness addresses no rule of Pol_C
    }
    gap = {k: [r["cid"] for r in rows if not f(r)] for k, f in sets.items()}
    # which rules C5 would discharge exactly, and the residue of their violations
    gateable = [r for r in rows if r["stepwise"]]
    single = [r for r in rows if r["call_head_gateable"]]
    res = {"rules": rows, "expressiveness_gap": gap,
           "n_gateable": len(gateable), "n_call_head_gateable": len(single),
           "violations_call_head_gateable": sum(r["viol"] for r in single),
           "n_forward": sum(r["forward"] and not r["stepwise"] for r in rows),
           "n_oracle_app": sum(r["oracle_applicability"] for r in rows),
           "violations_gateable": sum(r["viol"] for r in gateable),
           "violations_total": sum(r["viol"] for r in rows)}
    json.dump(res, open(out, "w"), indent=1)
    print(f"{'cid':<5}{'fwd':>5}{'loc':>5}{'orc':>5}{'step':>6}  {'App':>4}{'Viol':>6}  fragment")
    for r in rows:
        print(f"{r['cid']:<5}{str(r['forward']):>5}{str(r['local']):>5}{str(r['oracle_applicability']):>5}{str(r['stepwise']):>6}  {r['app']:>4}{r['viol']:>6}  {r['fragment']}")
    print({k: v for k, v in res.items() if k != 'rules'})

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/exp1.json", sys.argv[2] if len(sys.argv) > 2 else "out/exp2.json")
