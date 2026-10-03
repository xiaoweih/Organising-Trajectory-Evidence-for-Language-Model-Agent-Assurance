"""Check every count reported in the paper against the per-run results in out/.

Usage:  python3 verify_counts.py            (checks out/)
        python3 verify_counts.py OUTDIR     (checks a regenerated directory)

Checks rule and oracle totals from the 9,120 per-run rule verdicts, the suite
ledger, pairwise overlaps, flag unions, unique flags and the monitor table.
"""
from pathlib import Path
import json, sys
from itertools import combinations

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "out"


def main():
    if True:
        def read(name):
            return json.loads((OUT / (name + ".json")).read_text())

        e1, e2, e3, e3b, e3_B, e5 = (
            read(n) for n in ("exp1", "exp2", "exp3", "exp3b", "exp3_B", "exp5")
        )
        saved_monitor = read("monitor")["per_sim"]
    runs = e1["per_run"]
    assert len(runs) == 456
    assert len({r["sim_id"] for r in runs}) == len(runs)
    rules = {}
    for cid, summary in e1["summary"]["per_rule"].items():
        app = sum(bool(r["app"][cid]) for r in runs)
        viol = sum(bool(r["viol"][cid]) for r in runs)
        assert (app, viol) == (summary["app"], summary["viol"])
        assert summary["disagree"] == 0
        rules[cid] = {"applicable_runs": app, "violating_runs": viol,
                      "suite_k1": "established" if app >= 1 and viol == 0 else "not established"}
    established = [c for c, r in rules.items() if r["suite_k1"] == "established"]
    assert established == ["C01", "C02", "C04", "C06", "C08", "C10", "C16"]
    assert sum(r["violating_runs"] for r in rules.values()) == 992
    assert sum(r["violating_runs"] for c, r in rules.items() if c != "C11") == 540
    assert sum(r["applicable_runs"] >= 50 for r in rules.values()) == 18
    breaches = [r for r in runs if r["viol"]["C07"]]
    oracle_fail = sum(r["reward"] < 1 for r in runs)
    blind_spot = sum(r["reward"] == 1 for r in breaches)
    assert (oracle_fail, len(breaches), blind_spot) == (231, 30, 22)
    assert blind_spot == e5["c07_breaches"]["oracle_pass"]
    assert sum(r["blocked"] for r in runs) == 30
    assert sum(any(v for c, v in r["viol"].items() if c != "C11") for r in runs) == 338
    ids = {r["sim_id"] for r in runs}
    assert set(saved_monitor) == ids
    flags = {
        "C": {r["sim_id"] for r in runs if any(v for c, v in r["viol"].items() if c != "C11")},
        "G": {r["sim_id"] for r in runs if r["i4_viol"]},
        "M": {s for s, r in saved_monitor.items() if r["first_k"] is not None},
        "O": {r["sim_id"] for r in runs if r["reward"] < 1},
    }
    expected_both = {"CG": 102, "CM": 131, "CO": 178, "GM": 30, "GO": 46, "MO": 161}
    overlap = {}
    for a, b in combinations(flags, 2):
        both = len(flags[a] & flags[b])
        assert both == expected_both[a+b]
        overlap[a+b] = {"both": both, "expected_under_independence": len(flags[a])*len(flags[b])/len(runs),
                         "only_first": len(flags[a]-flags[b]), "only_second": len(flags[b]-flags[a]),
                         "disagreement": len(flags[a]^flags[b])/len(runs)}
    unique = {a: len(flags[a] - set.union(*(flags[b] for b in flags if b != a))) for a in flags}
    assert unique == {"C": 95, "G": 5, "M": 6, "O": 7}
    union = set.union(*flags.values())
    assert len(flags["C"] | flags["O"]) == 391
    assert len(flags["C"] | flags["G"] | flags["O"]) == 401
    assert len(flags["G"] | flags["M"] | flags["O"]) == 312
    assert len(union) == 407 and len(ids - union) == 49
    assert len(union | {r["sim_id"] for r in runs if r["viol"]["C11"]}) == 456
    assert len(flags["G"]) == 117 and len(flags["G"]-flags["O"]) == 71
    monitor = {}
    for label, expected in {"oracle": (183, 161, 0), "rules": (220, 112, 30), "c07": (64, 15, 22)}.items():
        row = e3["monitors"][label]
        observed = (row["n_alarmed"], row["detects"]["oracle"]["caught"], row["detects"]["c07"]["caught"])
        assert observed == expected
        monitor[label] = row
    assert [monitor[x]["detects"]["support"]["caught"] for x in ("oracle", "rules", "c07")] == [30, 84, 35]
    assert [e3_B["monitors"][x]["detects"]["c07"]["caught"] for x in ("oracle", "c07", "rules")] == [0, 20, 22]
    assert (e3b["resampled"]["caught_min"], e3b["resampled"]["caught_median"], e3b["resampled"]["caught_max"]) == (20, 24, 27)
    print("All paper counts verified against", OUT)


if __name__ == "__main__":
    main()
