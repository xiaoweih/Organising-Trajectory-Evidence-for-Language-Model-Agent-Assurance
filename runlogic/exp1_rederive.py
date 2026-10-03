"""E1: re-derive the pilot's numbers from formulas (faithfulness of the formalisation)."""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import adapter
from runlogic.logic import Checker
from runlogic.model import Run
from runlogic.rules_telecom import RULES, VIOL_I4, I4, I5, RESPONSE, SOUNDNESS, LACT_C07, guard_c07, FORBIDDEN, BLOCKED
from constraints import CONSTRAINTS
import grounding as GR

def main(path, out):
    traces = adapter.load(path)
    n = len(traces)
    per = {r.cid: {"app": 0, "viol": 0, "app_pilot": 0, "viol_pilot": 0, "disagree": 0} for r in RULES}
    runs = []
    per_run = []
    i4_viol = 0; g_pilot = 0; g_agree = 0
    gate = {"blocked": 0, "i5": 0, "response": 0, "sound": 0, "lact": 0}
    for tr in traces:
        r = Run(tr); c = Checker(r)
        row = {"sim_id": tr.sim_id, "reward": tr.reward, "app": {}, "viol": {}}
        for rule, con in zip(RULES, CONSTRAINTS):
            assert rule.cid == con.cid
            app = c.holds(rule.app); viol = c.holds(rule.viol)
            v = con.check(tr)
            papp = v is not None; pviol = v is False
            per[rule.cid]["app"] += app; per[rule.cid]["viol"] += viol
            per[rule.cid]["app_pilot"] += papp; per[rule.cid]["viol_pilot"] += pviol
            if (app, viol) != (papp, pviol): per[rule.cid]["disagree"] += 1
            row["app"][rule.cid] = app; row["viol"][rule.cid] = viol
        # I4 against the pilot's g1/g2
        iv = c.holds(VIOL_I4)
        s1, _ = GR.g1(tr); s2, _ = GR.g2(tr)
        pv = (s1 is False) or (s2 is False)
        i4_viol += iv; g_pilot += pv; g_agree += (iv == pv)
        row["i4_viol"] = iv
        # gate replay
        rg = Run(tr, guard=guard_c07); cg = Checker(rg)
        gate["blocked"] += len(rg.blocked)
        gate["i5"] += cg.holds(I5); gate["response"] += cg.holds(RESPONSE); gate["sound"] += cg.holds(SOUNDNESS)
        gate["lact"] += cg.holds(LACT_C07)
        row["blocked"] = len(rg.blocked)
        per_run.append(row)
    res = {"n": n, "per_rule": per, "i4": {"viol_runs": i4_viol, "pilot_unsupported_runs": g_pilot, "agree": g_agree},
           "gate": gate}
    json.dump({"summary": res, "per_run": per_run}, open(out, "w"))
    print(f"{n} runs")
    print(f"{'rule':<5}{'App':>6}{'Viol':>6}{'App*':>6}{'Viol*':>7}{'diff':>6}")
    for cid, d in per.items():
        print(f"{cid:<5}{d['app']:>6}{d['viol']:>6}{d['app_pilot']:>6}{d['viol_pilot']:>7}{d['disagree']:>6}")
    print("I4:", res["i4"]); print("gate:", gate)
    return res

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out/exp1.json")
