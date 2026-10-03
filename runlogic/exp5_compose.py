"""E5: composition on formula-derived verdicts: vote versus case, and lineage agreement."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HARD_EXCLUDE = {"C11"}
def main(exp1_path, monitor_path, out):
    e1 = json.load(open(exp1_path))["per_run"]; mon = json.load(open(monitor_path))["per_sim"]
    rows = []
    for r in e1:
        rules_clean = not any(v and c not in HARD_EXCLUDE for c, v in r["viol"].items())
        alarm = mon.get(r["sim_id"], {}).get("first_k") is not None
        rows.append({"ok": r["reward"] == 1.0, "rules": rules_clean, "support": not r["i4_viol"], "monitor": not alarm, "c07": r["viol"]["C07"]})
    n = len(rows)
    allclean = [r for r in rows if r["rules"] and r["support"] and r["monitor"]]
    anyflag = [r for r in rows if not (r["rules"] and r["support"] and r["monitor"])]
    res = {"n": n,
           "all_clean": {"n": len(allclean), "fail": sum(not r["ok"] for r in allclean)},
           "any_flag": {"n": len(anyflag), "succeed": sum(r["ok"] for r in anyflag)},
           "disagree_rules_monitor": sum(r["rules"] != r["monitor"] for r in rows) / n,
           "disagree_monitor_oracle": sum(r["monitor"] != r["ok"] for r in rows) / n,
           "disagree_rules_oracle": sum(r["rules"] != r["ok"] for r in rows) / n,
           "disagree_support_oracle": sum(r["support"] != r["ok"] for r in rows) / n,
           "c07_breaches": {"n": sum(r["c07"] for r in rows), "oracle_pass": sum(r["c07"] and r["ok"] for r in rows),
                            "monitor_silent": sum(r["c07"] and r["monitor"] for r in rows), "support_silent": sum(r["c07"] and r["support"] for r in rows)}}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
main(sys.argv[1], sys.argv[2], sys.argv[3])
