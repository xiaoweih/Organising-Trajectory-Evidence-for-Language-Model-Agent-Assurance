"""Behaviour constraints extracted from the tau2-bench telecom domain documents.

Each constraint is written in EARS-normalised form ("When <condition>, the agent
shall <observable behaviour>") and carries a deterministic checker over a
canonical trace. A checker returns:

    None  -> the applicability condition was not instantiated (UNCOVERED)
    True  -> instantiated and satisfied                        (COVERED / PASS)
    False -> instantiated and violated                         (COVERED / FAIL)

Applicability is judged from the trajectory alone, except where `oracle=True`,
which marks constraints whose applicability is read from the task's ground-truth
fault labels. Those are reported separately: an auditor has that information, a
deployed monitor does not.

Sources: data/tau2/domains/telecom/main_policy.md  (POLICY)
         data/tau2/domains/telecom/tech_support_workflow.md  (WORKFLOW)
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable
from adapter import Trace, Step

LOOKUP = {"get_customer_by_phone", "get_customer_by_id", "get_customer_by_name",
          "get_customer_by_name_dob"}
ACCOUNT_WRITES = {"make_payment", "send_payment_request", "refuel_data",
                  "resume_line", "suspend_line", "enable_roaming", "change_plan",
                  "apply_plan", "transfer_to_human_agents"}
# device-side actions that change state (performed by the user on the agent's guidance)
DEVICE_FIXES = {"toggle_airplane_mode", "reseat_sim_card", "toggle_data",
                "toggle_roaming", "toggle_data_saver_mode", "set_apn_settings",
                "reset_apn_settings", "toggle_wifi", "toggle_wifi_calling",
                "disconnect_vpn", "grant_app_permission", "reboot_device",
                "set_network_mode_preference"}
DIAGNOSTICS = {"check_status_bar", "check_network_status", "check_sim_status",
               "run_speed_test", "can_send_mms", "check_apn_settings",
               "check_network_mode_preference", "check_data_restriction_status",
               "check_wifi_status", "check_wifi_calling_status", "check_vpn_status",
               "check_app_permissions", "check_app_status", "check_installed_apps"}
TRANSFER_LINE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."


@dataclass
class Constraint:
    cid: str
    source: str
    ears: str
    check: Callable[[Trace], bool | None]
    oracle: bool = False          # applicability read from ground-truth faults


# ---------- helpers ---------------------------------------------------------

def _idx_of(t: Trace, tools: set[str], actor: str | None = None) -> int | None:
    for s in t.calls(actor):
        if s.tool in tools:
            return s.idx
    return None


def _last_idx_of(t: Trace, tools: set[str], actor: str | None = None) -> int | None:
    hits = [s.idx for s in t.calls(actor) if s.tool in tools]
    return hits[-1] if hits else None


def _details_calls(t: Trace, prefix: str) -> list[Step]:
    """get_details_by_id calls whose id has a given prefix (B=bill, L=line, P=plan)."""
    out = []
    for s in t.calls("assistant"):
        if s.tool == "get_details_by_id":
            i = str(s.args.get("id", ""))
            if i.upper().startswith(prefix):
                out.append(s)
    return out


DOMAIN_NOW = "2025-02-25"          # "The current time is ..." in main_policy.md
ABROAD_RE = re.compile(r"\babroad\b|\btravel(?:l?ing|s|ed)?\b|outside .{0,20}network|"
                       r"\bin (?:France|Spain|Italy|Germany|Japan|Canada|Mexico)\b", re.I)


def _line_payloads(t: Trace) -> dict[str, dict]:
    """Line records the agent actually observed, keyed by line id (last one wins)."""
    import json as _json
    out: dict[str, dict] = {}
    for s in t.calls("assistant"):
        if s.tool != "get_details_by_id":
            continue
        r = t.result_for(s)
        if r is None or r.error:
            continue
        try:
            payload = _json.loads(r.text)
        except Exception:
            continue
        if isinstance(payload, dict) and "line_id" in payload:
            out[str(payload["line_id"])] = payload
    return out


def _user_says_abroad(t: Trace) -> bool:
    return any(ABROAD_RE.search(s.text) for s in t.steps
               if s.kind == "message" and s.actor == "user")


def _bill_read_idx(t: Trace) -> int | None:
    cands = [s.idx for s in _details_calls(t, "B")]
    g = _idx_of(t, {"get_bills_for_customer"}, "assistant")
    if g is not None:
        cands.append(g)
    return min(cands) if cands else None


# ---------- POLICY constraints ---------------------------------------------

def c01(t: Trace):
    """Identify the customer before modifying the account."""
    writes = [s.idx for s in t.calls("assistant")
              if s.tool in ACCOUNT_WRITES and s.tool != "transfer_to_human_agents"]
    if not writes:
        return None
    look = _idx_of(t, LOOKUP, "assistant")
    return look is not None and look < min(writes)


def c02(t: Trace):
    """Check the bill status before sending a payment request."""
    req = _idx_of(t, {"send_payment_request"}, "assistant")
    if req is None:
        return None
    read = _bill_read_idx(t)
    return read is not None and read < req


def c03(t: Trace):
    """Re-check the bill after paying, before telling the user it is paid.

    make_payment is a user-side action in this domain, so applicability is
    judged over both actors.
    """
    pay = _last_idx_of(t, {"make_payment"})
    if pay is None:
        return None
    later = [s.idx for s in _details_calls(t, "B")]
    g = [s.idx for s in t.calls("assistant") if s.tool == "get_bills_for_customer"]
    return any(i > pay for i in later + g)


def c04(t: Trace):
    """Refuel at most 2 GB."""
    calls = [s for s in t.calls("assistant") if s.tool == "refuel_data"]
    if not calls:
        return None
    for s in calls:
        try:
            if float(s.args.get("gb_amount", 0)) > 2.0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def c05(t: Trace):
    """Obtain the plan's refuelling price before refuelling."""
    ref = _idx_of(t, {"refuel_data"}, "assistant")
    if ref is None:
        return None
    plan = [s.idx for s in _details_calls(t, "P")]
    return any(i < ref for i in plan)


def c06(t: Trace):
    """Read the line record before resuming a line."""
    res = _idx_of(t, {"resume_line"}, "assistant")
    if res is None:
        return None
    line = [s.idx for s in _details_calls(t, "L")]
    return any(i < res for i in line)


def c07(t: Trace):
    """Do not lift a suspension on a line whose contract end date has passed.

    Applicability is trajectory-grounded: the agent must itself have read a line
    record showing a suspended line with a past contract end date. The violation
    is a resume_line call naming that line.
    """
    expired = {lid for lid, p in _line_payloads(t).items()
               if str(p.get("contract_end_date") or "9999") < DOMAIN_NOW}
    if not expired:
        return None
    for s in t.calls("assistant"):
        if s.tool == "resume_line" and str(s.args.get("line_id", "")) in expired:
            return False
    return True


def c08(t: Trace):
    """After transferring, send exactly the mandated hand-off sentence."""
    tr = _idx_of(t, {"transfer_to_human_agents"}, "assistant")
    if tr is None:
        return None
    for s in t.steps[tr:]:
        if s.kind == "message" and s.actor == "assistant":
            return TRANSFER_LINE.lower() in s.text.strip().lower()
    return False


def c09(t: Trace):
    """Attempt the relevant resolution steps before transferring to a human."""
    tr = _idx_of(t, {"transfer_to_human_agents"}, "assistant")
    if tr is None:
        return None
    fixes = [s.idx for s in t.calls() if s.tool in DEVICE_FIXES
             or s.tool in (ACCOUNT_WRITES - {"transfer_to_human_agents"})]
    return any(i < tr for i in fixes)


def c10(t: Trace):
    """Make at most one tool call per message."""
    if not t.msg_call_counts:
        return None
    return max(t.msg_call_counts) <= 1


def c11(t: Trace):
    """Do not send prose and a tool call in the same message."""
    if not t.msg_call_counts:
        return None
    return t.mixed_msgs == 0


# ---------- WORKFLOW constraints -------------------------------------------

def c12(t: Trace):
    """For a no-service report, check the status bar before applying device fixes."""
    if t.issue_type != "service_issue":
        return None
    fix = _idx_of(t, DEVICE_FIXES)
    if fix is None:
        return None
    sb = _idx_of(t, {"check_status_bar"})
    return sb is not None and sb < fix


def c13(t: Trace):
    """Check network status (airplane mode) before checking the SIM."""
    sim = _idx_of(t, {"check_sim_status"})
    if sim is None:
        return None
    net = _idx_of(t, {"check_network_status"})
    return net is not None and net < sim


def c14(t: Trace):
    """After reseating the SIM, confirm the SIM is active or service restored."""
    rs = _last_idx_of(t, {"reseat_sim_card"})
    if rs is None:
        return None
    conf = [s.idx for s in t.calls() if s.tool in {"check_sim_status", "check_status_bar"}]
    return any(i > rs for i in conf)


def c15(t: Trace):
    """After the last device fix, re-run a diagnostic before concluding."""
    fix = _last_idx_of(t, DEVICE_FIXES)
    if fix is None:
        return None
    diag = [s.idx for s in t.calls() if s.tool in DIAGNOSTICS]
    return any(i > fix for i in diag)


def c16(t: Trace):
    """Check data usage against the plan limit before refuelling."""
    ref = _idx_of(t, {"refuel_data"}, "assistant")
    if ref is None:
        return None
    usage = [s.idx for s in t.calls("assistant")
             if s.tool == "get_data_usage"] + [s.idx for s in _details_calls(t, "L")]
    return any(i < ref for i in usage)


def c17(t: Trace):
    """When messaging permissions are at fault, check the app's permissions."""
    if not any(f.startswith("break_app_") for f in t.faults):
        return None
    return _idx_of(t, {"check_app_permissions", "check_app_status"}) is not None


def c18(t: Trace):
    """When data saver is at fault, check the data restriction settings."""
    if "data_saver_mode_on" not in t.faults:
        return None
    return _idx_of(t, {"check_data_restriction_status"}) is not None


def c19(t: Trace):
    """When the user is abroad and the line's roaming flag is off, enable it.

    Trajectory-grounded: the user says they are abroad and the agent has read a
    line record with roaming_enabled false. (An earlier version keyed this to the
    fault label `user_abroad_roaming_enabled_off`, which names the *device*
    toggle, not the line flag — see notes in the report.)
    """
    if not _user_says_abroad(t):
        return None
    off = [lid for lid, p in _line_payloads(t).items() if p.get("roaming_enabled") is False]
    if not off:
        return None
    return _idx_of(t, {"enable_roaming"}, "assistant") is not None


def c20(t: Trace):
    """When the network mode preference is at fault, read it before changing it."""
    if "bad_network_preference" not in t.faults:
        return None
    setp = _idx_of(t, {"set_network_mode_preference"})
    if setp is None:
        return False
    chk = _idx_of(t, {"check_network_mode_preference"})
    return chk is not None and chk < setp


CONSTRAINTS: list[Constraint] = [
    Constraint("C01", "POLICY §Customer Lookup",
               "When the agent modifies the account, it shall first identify the customer.", c01),
    Constraint("C02", "POLICY §Overdue Bill Payment",
               "When the agent sends a payment request, it shall first check that the bill is overdue.", c02),
    Constraint("C03", "POLICY §Overdue Bill Payment",
               "When a payment has been made, the agent shall verify the bill status before confirming payment.", c03),
    Constraint("C04", "POLICY §Data Refueling",
               "When the agent refuels data, the amount shall not exceed 2 GB.", c04),
    Constraint("C05", "POLICY §Data Refueling",
               "When the agent refuels data, it shall first obtain the plan's refuelling price.", c05),
    Constraint("C06", "POLICY §Line Suspension",
               "When the agent resumes a line, it shall first read the line record.", c06),
    Constraint("C07", "POLICY §Line Suspension",
               "When a line's contract end date has passed, the agent shall not lift the suspension.", c07),
    Constraint("C08", "POLICY §Transfer",
               "When the agent transfers to a human, it shall send the mandated hand-off sentence.", c08),
    Constraint("C09", "WORKFLOW §Introduction",
               "When the agent transfers to a human, it shall first attempt the relevant resolution steps.", c09),
    Constraint("C10", "POLICY §Preamble",
               "When the agent acts, it shall make at most one tool call per message.", c10),
    Constraint("C11", "POLICY §Preamble",
               "When the agent makes a tool call, it shall not also reply to the user in that message.", c11),
    Constraint("C12", "WORKFLOW §Step 1.0",
               "When the user reports no service, the agent shall check the status bar before applying fixes.", c12),
    Constraint("C13", "WORKFLOW §Steps 1.1-1.2",
               "When the agent checks the SIM, it shall first have checked network status.", c13),
    Constraint("C14", "WORKFLOW §Step 1.2",
               "When the SIM has been reseated, the agent shall confirm the SIM is active or service restored.", c14),
    Constraint("C15", "WORKFLOW §Paths 1-3",
               "When a device fix has been applied, the agent shall re-run a diagnostic before concluding.", c15),
    Constraint("C16", "WORKFLOW §Step 2.1.4",
               "When the agent refuels data, it shall first check data usage against the limit.", c16),
    Constraint("C17", "WORKFLOW §Step 3.5",
               "When messaging permissions are implicated, the agent shall check the app's permissions.", c17, oracle=True),
    Constraint("C18", "WORKFLOW §Step 2.2.1",
               "When data saver is implicated, the agent shall check data restriction settings.", c18, oracle=True),
    Constraint("C19", "POLICY §Data Roaming",
               "When the user is abroad and line roaming is not enabled, the agent shall enable it.", c19),
    Constraint("C20", "WORKFLOW §Step 2.2.2",
               "When the network mode preference is implicated, the agent shall read it before changing it.", c20, oracle=True),
]
