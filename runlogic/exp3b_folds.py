"""E3b: fold-level and resampling detail for the c7-trained monitor of E3.
Same features, model and threshold rule as exp3_relabel.py; reports (i) per-fold caught/positives
for the deterministic GroupKFold split and (ii) the spread of total caught over 20 random
task-to-fold assignments."""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
import adapter, monitor as MON
from runlogic.exp3_relabel import labels_from, fit_oof, run_alarms

def evaluate(texts, y, groups, meta, sims, lab, tgt):
    oof = fit_oof(texts, y, groups)
    meta_l = [dict(m, failed=bool(lab[m["sim_id"]])) for m in meta]
    thr = MON.threshold_for_far(meta_l, oof, 0.10)
    first = run_alarms(meta, oof, thr)
    pos = [s for s in sims if tgt[s]]
    return oof, first, sum(s in first for s in pos), len(pos), len(first)

def main(path, exp1_path, out, src="c07", tgt="c07"):
    traces = adapter.load(path); exp1 = json.load(open(exp1_path))
    texts, _, groups, meta = MON.build_dataset(traces)
    L = labels_from(exp1, traces); sims = [t.sim_id for t in traces]
    lab = L[src]; y = np.array([lab[m["sim_id"]] for m in meta])
    res = {"src": src, "tgt": tgt}
    # (i) per fold, deterministic split
    oof, first, caught, of, n_al = evaluate(texts, y, groups, meta, sims, lab, L[tgt])
    gkf = GroupKFold(n_splits=5); folds = []
    sim_of = {}
    for i, m in enumerate(meta): sim_of.setdefault(m["sim_id"], i)
    for k, (tr, te) in enumerate(gkf.split(texts, y, groups)):
        te_sims = {meta[i]["sim_id"] for i in te}
        pos = [s for s in te_sims if L[tgt][s]]
        folds.append({"fold": k, "test_runs": len(te_sims), "positives": len(pos),
                      "caught": sum(s in first for s in pos), "alarmed": sum(s in first for s in te_sims)})
    res["deterministic"] = {"caught": caught, "of": of, "alarmed": n_al, "folds": folds}
    # (ii) random task-to-fold assignments
    rng = np.random.default_rng(0); uniq = sorted(set(groups)); tot = []
    for r in range(20):
        perm = {g: int(v) for g, v in zip(uniq, rng.permutation(len(uniq)))}
        g2 = [perm[g] for g in groups]
        _, _, c, o, na = evaluate(texts, y, g2, meta, sims, lab, L[tgt])
        tot.append({"caught": c, "alarmed": na})
    cs = [t["caught"] for t in tot]
    res["resampled"] = {"n": len(tot), "caught_min": min(cs), "caught_median": float(np.median(cs)), "caught_max": max(cs),
                        "alarmed_median": float(np.median([t["alarmed"] for t in tot])), "runs": tot}
    print(json.dumps({k: v for k, v in res.items() if k != "resampled"}, indent=1))
    print("resampled caught:", cs)
    json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main(*sys.argv[1:4])
