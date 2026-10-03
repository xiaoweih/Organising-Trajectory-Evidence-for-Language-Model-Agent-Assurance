# Artifact: Organising Trajectory Evidence for Language-Model Agent Assurance

Code, rules and per-run results for the paper's evaluation on the released
tau2-bench telecom trajectories. Everything in the paper's evaluation section can be
regenerated from public data with one command; no model calls are needed.

## Contents

| Path | What it is |
|---|---|
| `rules.md` | The twenty audited rules with source spans and per-rule counts (paper Table 2). |
| `runlogic/logic.py` | The two-tier logic: syntax, finite-trace semantics, fragment membership. |
| `runlogic/model.py` | Trajectory model: atoms, event predicates, deterministic argument structures, gate replay. |
| `runlogic/rules_telecom.py` | The rules as formulas, the support ideal, the gate guard for rule C07. |
| `constraints.py` | Second, separately written procedural implementation of the same rules. |
| `adapter.py` | Loader for tau2-bench result files. |
| `grounding.py` | The support stand-in: deterministic support predicates on the agent's prefix. |
| `monitor.py` | The prefix-monitor baseline (TF-IDF features, class-balanced logistic regression, task-grouped folds). |
| `runlogic/exp1_rederive.py` | Rule checker, cross-check against `constraints.py`, support stand-in and gate replay on every run. |
| `runlogic/exp2_fragments.py` | Fragment membership of each rule and expressiveness gaps. |
| `runlogic/exp3_relabel.py`, `exp3b_folds.py` | The monitor trained on three label sources; fold sensitivity. |
| `runlogic/exp4_change.py` | Claims under the second released configuration. |
| `runlogic/exp5_compose.py` | Overlaps, unions and composition of the detection sources. |
| `out/` | Saved per-run verdicts and summaries used in the paper. |
| `verify_counts.py` | Checks every count reported in the paper against `out/` (or a regenerated directory). |
| `reproduce.sh` | Regenerates `out/` from scratch and runs the checks. |

## Data

The trajectories are not redistributed here. They are part of the public tau2-bench
repository (https://github.com/sierra-research/tau2-bench), directory
`data/tau2/results/final/`:

- main configuration: `claude-3-7-sonnet-20250219_telecom_default_gpt-4.1-2025-04-14_4trials.json`
  (assistant `claude-3-7-sonnet-20250219`, simulated customer `gpt-4.1-2025-04-14`, 114 tasks x 4 trials = 456 runs);
  SHA-256 `f49e540896fe91ab8631647f02eb777ef6fed5a6ebec545f43e71d0504e227b2`;
- second configuration: `gpt-4.1-2025-04-14_telecom-workflow_default_gpt-4.1-2025-04-14_4trials.json`.

Results were reproduced at repository commit `5bfa7e37b36656b37dc6d022156be6563c1007f3`;
the main trajectory file was last changed at commit `e10cffa7512157c157a29f71e5de4286cef5acb7`.

## Usage

Requires Python 3.10+ and `pip install -r requirements.txt` (tested with Python 3.11,
numpy 2.4, scikit-learn 1.8).

```sh
python3 verify_counts.py        # check the paper's counts against the saved results (seconds)
./reproduce.sh                  # clone tau2-bench, regenerate everything into regen/, verify (about 8 minutes)
./reproduce.sh /path/to/tau2-bench   # use an existing checkout
```

`reproduce.sh` regenerates every file in `out/`; the regenerated files are identical to
the saved ones, and `verify_counts.py` then checks the paper's numbers on them.

## Scope

The support checker and the monitor are stand-ins for the methods cited in the paper,
not reimplementations of them. The rules are one audited reading of the policy; the
two implementations share that reading, so their agreement checks the code, not the
interpretation.
