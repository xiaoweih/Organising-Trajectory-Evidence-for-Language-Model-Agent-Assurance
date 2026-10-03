"""E3: lineage as a measured effect. Train the stand-in prefix monitor on different label
sources (oracle; rule check; C07 only; support check) under the same task-disjoint folds,
and measure which failure events each monitor detects at run-level FAR <= 10%."""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
import adapter, monitor as MON

HARD_EXCLUDE = {"C11"}   # formatting rule, violated in 452/456 runs; excluded as in compose.py

def labels_from(exp1, traces):
    per = {r["sim_id"]: r for r in exp1["per_run"]}
    L = {}
    L["oracle"] = {t.sim_id: int(t.failed) for t in traces}
    L["rules"] = {s: int(any(v and c not in HARD_EXCLUDE for c, v in per[s]["viol"].items())) for s in per}
    L["c07"] = {s: int(per[s]["viol"]["C07"]) for s in per}
    L["support"] = {s: int(per[s]["i4_viol"]) for s in per}
    L["rules_or_oracle"] = {s: int(L["rules"][s] or L["oracle"][s]) for s in per}
    return L

def fit_oof(texts, y, groups, folds=5):
    gkf = GroupKFold(n_splits=folds); oof = np.zeros(len(y))
    for tr, te in gkf.split(texts, y, groups):
        if y[tr].sum() == 0 or y[tr].sum() == len(tr):
            oof[te] = y[tr].mean(); continue
        clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True),
                            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
        clf.fit([texts[i] for i in tr], y[tr]); oof[te] = clf.predict_proba([texts[i] for i in te])[:, 1]
    return oof

def run_alarms(meta, scores, thr):
    first = {}
    for m, s in zip(meta, scores):
        if s >= thr and (m["sim_id"] not in first or m["k"] < first[m["sim_id"]]): first[m["sim_id"]] = m["k"]
    return first

def main(path, exp1_path, out):
    traces = adapter.load(path); exp1 = json.load(open(exp1_path))
    texts, _, groups, meta = MON.build_dataset(traces)
    L = labels_from(exp1, traces)
    sims = [t.sim_id for t in traces]
    res = {"n_positive": {k: int(sum(v.values())) for k, v in L.items()}, "monitors": {}}
    for src, lab in L.items():
        y = np.array([lab[m["sim_id"]] for m in meta])
        oof = fit_oof(texts, y, groups)
        ap = average_precision_score(y, oof) if 0 < y.mean() < 1 else None
        # threshold at 10% FAR w.r.t. the *training* label's negatives
        meta_l = [dict(m, failed=bool(lab[m["sim_id"]])) for m in meta]
        thr = MON.threshold_for_far(meta_l, oof, 0.10)
        first = run_alarms(meta, oof, thr)
        row = {"auprc": ap, "base": float(y.mean()), "n_alarmed": len(first), "detects": {}}
        for tgt, tl in L.items():
            pos = [s for s in sims if tl[s]]
            row["detects"][tgt] = {"caught": sum(s in first for s in pos), "of": len(pos)}
        res["monitors"][src] = row
        print(f"labels={src:<16} AUPRC={ap if ap is None else round(ap,3)} base={y.mean():.3f} alarmed={len(first)}  " +
              " ".join(f"{k}:{v['caught']}/{v['of']}" for k, v in row["detects"].items()))
    json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
