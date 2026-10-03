"""Early warning: can an observed prefix tell us the run is going to fail?

Setting follows the PrefixGuard framing (learn from execution prefixes and
terminal outcomes, no judge in the loop at deployment) but the model here is a
simple TF-IDF + logistic regression baseline, not their architecture.

Two things are reported, because they answer different questions:
  * AUPRC over prefixes           -- can it rank failing prefixes above others?
  * recall / lead time at a fixed -- would a deployed alarm actually warn in
    trajectory-level false alarm      time to do something?
"""
from __future__ import annotations
import os, json, sys
import numpy as np
OUTDIR = os.environ.get("OUT", "out")
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
import adapter

MIN_PREFIX = 4          # ignore the opening exchange
STRIDE = int(os.environ.get("STRIDE", "2"))
WINDOW = 6              # full detail for the most recent steps only


def _event(s) -> str:
    """A coarse event symbol for a step: what happened, not what was said."""
    if s.kind == "tool_call":
        return f"CALL_{s.tool}"
    if s.kind == "tool_result":
        return f"ERR_{s.tool}" if s.error else f"OK_{s.tool}"
    return f"MSG_{s.actor}"


def build_dataset(traces):
    """One row per prefix. Features are computable online; no trajectory totals.

    A prefix is represented by its event history (a token stream, so repetition
    and ordering are visible to n-grams) plus verbatim text for the last WINDOW
    steps. This keeps the representation linear in prefix length.
    """
    texts, y, groups, meta = [], [], [], []
    for t in traces:
        steps = t.steps
        n = len(steps)
        events = [_event(s) for s in steps]
        serial = [s.serialize()[:200] for s in steps]
        for k in range(MIN_PREFIX, n + 1, STRIDE):
            head = " ".join(events[:k])
            tail = " \n".join(serial[max(0, k - WINDOW):k])
            texts.append(f"{head} || {tail}")
            y.append(1 if t.failed else 0)
            groups.append(t.task_id)
            meta.append({"sim_id": t.sim_id, "k": k, "n": n, "task_id": t.task_id,
                         "failed": bool(t.failed)})
    return texts, np.array(y), np.array(groups), meta


def trajectory_alarm_metrics(meta, scores, threshold):
    """Trajectory-level alarm behaviour at a given prefix-score threshold."""
    by_sim = defaultdict(list)
    for m, s in zip(meta, scores):
        by_sim[m["sim_id"]].append((m["k"], m["n"], m["failed"], s))
    fired_fail, fired_ok, leads, n_of = 0, 0, [], []
    n_fail = n_ok = 0
    for sim, rows in by_sim.items():
        rows.sort()
        failed = rows[0][2]
        n = rows[0][1]
        first = next((k for k, _, _, s in rows if s >= threshold), None)
        if failed:
            n_fail += 1
            if first is not None:
                fired_fail += 1
                leads.append((n - first) / n)      # fraction of the run still to come
                n_of.append(n)
        else:
            n_ok += 1
            if first is not None:
                fired_ok += 1
    return {"recall": fired_fail / max(n_fail, 1),
            "false_alarm": fired_ok / max(n_ok, 1),
            "median_lead": float(np.median(leads)) if leads else 0.0,
            # fraction of *detected* failing runs alarmed with more than 25% of the run remaining
            "early_recall_among_detected": float(np.mean([l > 0.25 for l in leads])) if leads else 0.0,
            # the early-warning recall r of the paper: over *all* failing runs
            "early_recall": sum(l > 0.25 for l in leads) / max(n_fail, 1),
            "early_recall_lead3": sum(l * n_of[i] >= 3 for i, l in enumerate(leads)) / max(n_fail, 1)}


def threshold_for_far(meta, scores, target_far=0.10):
    """Highest threshold whose trajectory-level false-alarm rate is <= target."""
    lo, hi = float(np.min(scores)), float(np.max(scores))
    best = hi
    for _ in range(40):
        mid = (lo + hi) / 2
        far = trajectory_alarm_metrics(meta, scores, mid)["false_alarm"]
        if far > target_far:
            lo = mid
        else:
            best, hi = mid, mid
    return best


def run(path, folds=5, seed=0, loader=None):
    traces = (loader or adapter.load)(path)
    texts, y, groups, meta = build_dataset(traces)
    print(f"prefixes: {len(texts)}  positives: {y.mean():.3f}  "
          f"tasks: {len(set(groups))}  trajectories: {len(traces)}")

    gkf = GroupKFold(n_splits=folds)
    oof = np.zeros(len(y))
    for fold, (tr, te) in enumerate(gkf.split(texts, y, groups), 1):
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000,
                            sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
        clf.fit([texts[i] for i in tr], y[tr])
        oof[te] = clf.predict_proba([texts[i] for i in te])[:, 1]
        print(f"  fold {fold}: train {len(tr)} test {len(te)}  "
              f"AP={average_precision_score(y[te], oof[te]):.3f}")

    ap = average_precision_score(y, oof)
    base = y.mean()
    print(f"\nprefix AUPRC (out-of-fold, task-disjoint): {ap:.3f}   baseline {base:.3f}")

    # a length-only control: how much is predictable from "this run is dragging on"?
    ln = np.array([[m["k"]] for m in meta], dtype=float)
    oof_len = np.zeros(len(y))
    for tr, te in gkf.split(ln, y, groups):
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(ln[tr], y[tr])
        oof_len[te] = lr.predict_proba(ln[te])[:, 1]
    print(f"prefix AUPRC, prefix-length only          : "
          f"{average_precision_score(y, oof_len):.3f}")

    # Control: is this predicting deterioration, or just how hard the task looked
    # at the start? Score every prefix by a model that sees only the opening.
    first_of = {}
    for i, m in enumerate(meta):
        j = first_of.get(m["sim_id"])
        if j is None or m["k"] < meta[j]["k"]:
            first_of[m["sim_id"]] = i
    open_idx = np.array(sorted(first_of.values()))
    oof_open = np.zeros(len(y))
    for tr, te in gkf.split(texts, y, groups):
        tr_open = np.intersect1d(tr, open_idx)
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                            sublinear_tf=True),
            LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit([texts[i] for i in tr_open], y[tr_open])
        oof_open[te] = clf.predict_proba([texts[i] for i in te])[:, 1]
    print(f"prefix AUPRC, opening-only model          : "
          f"{average_precision_score(y, oof_open):.3f}   "
          f"(how much is task difficulty, visible before anything goes wrong)")

    print("\nDeployment operating points (trajectory level):")
    rows = []
    for far in (0.05, 0.10, 0.20):
        thr = threshold_for_far(meta, oof, far)
        m = trajectory_alarm_metrics(meta, oof, thr)
        rows.append({"target_far": far, **m})
        print(f"  FAR<={far:.0%}: recall {m['recall']:.3f}  actual FAR {m['false_alarm']:.3f}  "
              f"median lead {m['median_lead']:.2f} of run  "
              f"early (>25% left) {m['early_recall']:.3f}")

    per_sim = {}
    for m, s in zip(meta, oof):
        d = per_sim.setdefault(m["sim_id"], {"task_id": m["task_id"],
                                             "failed": m["failed"], "max": 0.0,
                                             "n": m["n"], "first_k": None})
        d["max"] = max(d["max"], float(s))
    thr10 = threshold_for_far(meta, oof, 0.10)
    for m, s in zip(meta, oof):
        d = per_sim[m["sim_id"]]
        if s >= thr10 and (d["first_k"] is None or m["k"] < d["first_k"]):
            d["first_k"] = m["k"]
    # raw per-prefix out-of-fold scores, for the rare-event analysis
    np.savez(OUTDIR + "/prefix_scores.npz",
             score=oof,
             k=np.array([m["k"] for m in meta]),
             n=np.array([m["n"] for m in meta]),
             failed=np.array([m["failed"] for m in meta]),
             sim=np.array([m["sim_id"] for m in meta]),
             task=np.array([m["task_id"] for m in meta]))

    with open(OUTDIR + "/monitor.json", "w") as fh:
        json.dump({"auprc": ap, "baseline": base, "operating_points": rows,
                   "threshold_far10": thr10, "per_sim": per_sim}, fh)
    print("\nwrote out/monitor.json")
    return per_sim


if __name__ == "__main__":
    path = sys.argv[1]
    if path.endswith(".parquet"):
        import swe_adapter
        run(path, loader=swe_adapter.load)
    else:
        run(path)
