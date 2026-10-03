# The twenty audited rules

Each rule has the form *when rho, the agent shall chi*. `Source` is the section of the tau2-bench telecom policy (`POLICY`) or troubleshooting workflow (`WORKFLOW`) it was extracted from. Counts are over the 456 runs of the main configuration (`out/exp1.json`). Rules marked * read task fault metadata available to the assessor but not to the agent. Executable formulas: `runlogic/rules_telecom.py`; independent procedural implementation: `constraints.py`.

| Rule | Source | Requirement | Applicable | Violated |
|---|---|---|---:|---:|
| C01 | POLICY §Customer Lookup | When the agent modifies the account, it shall first identify the customer. | 302 | 0 |
| C02 | POLICY §Overdue Bill Payment | When the agent sends a payment request, it shall first check that the bill is overdue. | 84 | 0 |
| C03 | POLICY §Overdue Bill Payment | When a payment has been made, the agent shall verify the bill status before confirming payment. | 83 | 2 |
| C04 | POLICY §Data Refueling | When the agent refuels data, the amount shall not exceed 2 GB. | 181 | 0 |
| C05 | POLICY §Data Refueling | When the agent refuels data, it shall first obtain the plan's refuelling price. | 181 | 4 |
| C06 | POLICY §Line Suspension | When the agent resumes a line, it shall first read the line record. | 83 | 0 |
| C07 | POLICY §Line Suspension | When a line's contract end date has passed, the agent shall not lift the suspension. | 32 | 30 |
| C08 | POLICY §Transfer | When the agent transfers to a human, it shall send the mandated hand-off sentence. | 269 | 0 |
| C09 | WORKFLOW §Introduction | When the agent transfers to a human, it shall first attempt the relevant resolution steps. | 269 | 2 |
| C10 | POLICY §Preamble | When the agent acts, it shall make at most one tool call per message. | 456 | 0 |
| C11 | POLICY §Preamble | When the agent makes a tool call, it shall not also reply to the user in that message. | 456 | 452 |
| C12* | WORKFLOW §Step 1.0 | When the user reports no service, the agent shall check the status bar before applying fixes. | 113 | 92 |
| C13 | WORKFLOW §Steps 1.1-1.2 | When the agent checks the SIM, it shall first have checked network status. | 47 | 18 |
| C14 | WORKFLOW §Step 1.2 | When the SIM has been reseated, the agent shall confirm the SIM is active or service restored. | 218 | 118 |
| C15 | WORKFLOW §Paths 1-3 | When a device fix has been applied, the agent shall re-run a diagnostic before concluding. | 435 | 23 |
| C16 | WORKFLOW §Step 2.1.4 | When the agent refuels data, it shall first check data usage against the limit. | 181 | 0 |
| C17* | WORKFLOW §Step 3.5 | When messaging permissions are implicated, the agent shall check the app's permissions. | 140 | 37 |
| C18* | WORKFLOW §Step 2.2.1 | When data saver is implicated, the agent shall check data restriction settings. | 76 | 46 |
| C19 | POLICY §Data Roaming | When the user is abroad and line roaming is not enabled, the agent shall enable it. | 93 | 36 |
| C20* | WORKFLOW §Step 2.2.2 | When the network mode preference is implicated, the agent shall read it before changing it. | 224 | 132 |

C11 (formatting) is excluded from the substantive totals in the paper: 540 violated rule-run pairs over the other nineteen rules, 992 including C11.
