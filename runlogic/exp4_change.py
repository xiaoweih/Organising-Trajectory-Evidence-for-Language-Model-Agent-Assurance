"""E4: configuration change. Same policy, same tasks, a different model (Sigma_A -> Sigma_B).
Which rule-level claims survive?  A claim C1(c) with k=1 'survives' if it holds in both."""
import sys, os, json
def status(d, k=1):
    return "met" if d["app"] >= k and d["viol"] == 0 else ("violated" if d["viol"] > 0 else "unreached")
def main(a, b, out):
    A = json.load(open(a))["summary"]; B = json.load(open(b))["summary"]
    rows = []
    for cid in A["per_rule"]:
        sa, sb = status(A["per_rule"][cid]), status(B["per_rule"][cid])
        rows.append({"cid": cid, "A": (A["per_rule"][cid]["app"], A["per_rule"][cid]["viol"], sa),
                     "B": (B["per_rule"][cid]["app"], B["per_rule"][cid]["viol"], sb),
                     "survives": sa == sb == "met"})
    summ = {"met_A": sum(r["A"][2] == "met" for r in rows), "met_B": sum(r["B"][2] == "met" for r in rows),
            "survive": sum(r["survives"] for r in rows), "changed": [r["cid"] for r in rows if r["A"][2] != r["B"][2]],
            "i4": (A["i4"], B["i4"]), "gate": (A["gate"], B["gate"])}
    json.dump({"rows": rows, "summary": summ}, open(out, "w"), indent=1)
    for r in rows: print(r["cid"], r["A"], r["B"], "survives" if r["survives"] else "")
    print(summ)
main(sys.argv[1], sys.argv[2], sys.argv[3])
