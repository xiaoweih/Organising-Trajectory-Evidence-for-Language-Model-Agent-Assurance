"""The twenty telecom rules as formulas c = (rho_c, chi_c) of the logic.

Each rule is G(rho -> chi).  App_c = F rho, Viol_c = F(rho & ~chi).
The rules mirror constraints.py (the pilot's hand-written checkers) so that the
re-derivation experiment can compare them one by one.
"""
from __future__ import annotations
from .logic import *
from .model import *
import constraints as K
import grounding as GR

# ---- shared atoms ------------------------------------------------------------
WRITE = proposed_any(K.ACCOUNT_WRITES - {"transfer_to_human_agents"}, "proposed(accountWrite)")
LOOKUP = proposed_any(K.LOOKUP, "proposed(lookup)")
BILLREAD = Or(details("B", "proposed(billDetails)"), proposed("get_bills_for_customer"))
PLANREAD = details("P", "proposed(planDetails)")
LINEREAD = details("L", "proposed(lineDetails)")
PAYREQ = proposed("send_payment_request")
PAY = called_any({"make_payment"}, "proposed(make_payment,either)")     # user-side action in this domain
REFUEL = proposed("refuel_data")
RESUME = proposed("resume_line")
TRANSFER = proposed("transfer_to_human_agents")
FIX = called_any(K.DEVICE_FIXES | (K.ACCOUNT_WRITES - {"transfer_to_human_agents"}), "proposed(fix,either)")
DEVFIX = called_any(K.DEVICE_FIXES, "proposed(deviceFix,either)")
DIAG = called_any(K.DIAGNOSTICS, "proposed(diagnostic,either)")
STATUSBAR = called_any({"check_status_bar"}, "proposed(check_status_bar,either)")
NETSTAT = called_any({"check_network_status"}, "proposed(check_network_status,either)")
SIMSTAT = called_any({"check_sim_status"}, "proposed(check_sim_status,either)")
RESEAT = called_any({"reseat_sim_card"}, "proposed(reseat_sim_card,either)")
CONFIRM_SIM = called_any({"check_sim_status", "check_status_bar"}, "proposed(simConfirm,either)")
USAGEREAD = Or(proposed("get_data_usage"), LINEREAD)
APPCHECK = called_any({"check_app_permissions", "check_app_status"}, "proposed(appCheck,either)")
DATARESTR = called_any({"check_data_restriction_status"}, "proposed(dataRestriction,either)")
ENABLEROAM = proposed("enable_roaming")
SETNET = called_any({"set_network_mode_preference"}, "proposed(setNetMode,either)")
CHKNET = called_any({"check_network_mode_preference"}, "proposed(checkNetMode,either)")
AMSG = assistant_msg()
TRANSFER_LINE_MSG = assistant_msg(lambda s: K.TRANSFER_LINE.lower() in s.text.strip().lower(), "assistantMsg(transferLine)")
ANYCALL = Atom("proposed(any)", lambda r, t: r.proposed(t))
INIT = Atom("init", lambda r, t: t == 0)

def _multi(r, t):
    s = r.steps[t]
    if not (s.kind == "tool_call" and s.actor == "assistant"): return False
    for j in (t - 1, t + 1):
        if 0 <= j < len(r.steps):
            o = r.steps[j]
            if o.kind == "tool_call" and o.actor == "assistant" and o.turn_idx == s.turn_idx: return True
    return False
MULTICALL = Atom("multiCallMessage", _multi)

def _mixed(r, t):
    s = r.steps[t]
    if not (s.kind == "tool_call" and s.actor == "assistant") or t == 0: return False
    p = r.steps[t - 1]
    return p.kind == "message" and p.actor == "assistant" and p.turn_idx == s.turn_idx
MIXED = Atom("proseAndCall", _mixed)

# task-metadata atoms: known to the assessor from the task record, not to the assistant
ISSUE_SERVICE = domain("issue(service)", lambda r, t: r.issue_type == "service_issue", local=False)
FAULT_APP = domain("fault(app)", lambda r, t: any(f.startswith("break_app_") for f in r.faults), local=False)
FAULT_SAVER = domain("fault(dataSaver)", lambda r, t: "data_saver_mode_on" in r.faults, local=False)
FAULT_NET = domain("fault(netPref)", lambda r, t: "bad_network_preference" in r.faults, local=False)
ABROAD = domain("userSaidAbroad", lambda r, t: r.user_said_abroad(t))

# data-domain quantification over line ids the assistant has observed
def LINES(r, t): return list(r.line_records(t).keys())
def expired(l): return domain(f"contractEnded({l})", lambda r, t: l in r.expired_lines(t))
def roaming_off(l): return domain(f"roamingOff({l})", lambda r, t: l in r.roaming_off_lines(t))
def resume_of(l): return executed("resume_line", lambda s: str(s.args.get("line_id", "")) == l, f"executed(resume_line({l}))")

def refuel_gt2(s):
    try: return float(s.args.get("gb_amount", 0)) > 2.0
    except (TypeError, ValueError): return True
REFUEL_GT2 = proposed("refuel_data", refuel_gt2, "proposed(refuel>2GB)")

# ---- the rules ------------------------------------------------------------------
class Rule:
    def __init__(self, cid, rho, chi, text, oracle=False):
        self.cid, self.rho, self.chi, self.text, self.oracle = cid, rho, chi, text, oracle
    @property
    def formula(self): return G(self.rho >> self.chi)
    @property
    def app(self): return F(self.rho)
    @property
    def viol(self): return F(self.rho & ~self.chi)

RULES = [
    Rule("C01", WRITE, O(LOOKUP), "identify the customer before modifying the account"),
    Rule("C02", PAYREQ, O(BILLREAD), "check the bill before sending a payment request"),
    Rule("C03", PAY, F(BILLREAD) & ~BILLREAD, "re-check the bill after a payment"),
    Rule("C04", REFUEL, ~REFUEL_GT2, "refuel at most 2 GB"),
    Rule("C05", REFUEL, O(PLANREAD) & ~PLANREAD, "obtain the plan price before refuelling"),
    Rule("C06", RESUME, O(LINEREAD) & ~LINEREAD, "read the line record before resuming"),
    Rule("C07", Exists("l", lambda l: expired(l), LINES),
               Forall("l", lambda l: expired(l) >> G(~resume_of(l)), LINES),
               "never restore a line whose contract has ended"),
    Rule("C08", TRANSFER, X(U(~AMSG, TRANSFER_LINE_MSG)), "send the mandated hand-off sentence after transferring"),
    Rule("C09", TRANSFER, O(FIX), "attempt resolution steps before transferring"),
    Rule("C10", ANYCALL, ~MULTICALL, "at most one tool call per message"),
    Rule("C11", ANYCALL, ~MIXED, "no prose and a tool call in one message"),
    Rule("C12", ISSUE_SERVICE & DEVFIX, O(STATUSBAR) & ~STATUSBAR, "check the status bar before device fixes (no-service report)"),
    Rule("C13", SIMSTAT, O(NETSTAT) & ~NETSTAT, "check network status before the SIM"),
    Rule("C14", RESEAT, X(F(CONFIRM_SIM)), "confirm the SIM after reseating"),
    Rule("C15", DEVFIX, X(F(DIAG)), "re-run a diagnostic after a device fix"),
    Rule("C16", REFUEL, O(USAGEREAD) & ~USAGEREAD, "check usage before refuelling"),
    Rule("C17", FAULT_APP & INIT, F(APPCHECK), "check app permissions when messaging permissions are at fault", oracle=True),
    Rule("C18", FAULT_SAVER & INIT, F(DATARESTR), "check data restriction when data saver is at fault", oracle=True),
    Rule("C19", ABROAD & Exists("l", lambda l: roaming_off(l), LINES), O(ENABLEROAM) | F(ENABLEROAM), "enable roaming when abroad and the line's roaming is off"),
    Rule("C20", FAULT_NET & INIT, F(SETNET) & G(SETNET >> (O(CHKNET) & ~CHKNET)), "read the network mode before changing it", oracle=True),
]

# ---- the support ideal I4 and the gate --------------------------------------------
CHECKED = Atom("checked", lambda r, t: r.checked(t))
def grounded(alpha): return ArgBridge(alpha, GROUNDED_CHI)
I4_BODY = Forall("alpha", lambda a: CHECKED >> grounded(a))
I4 = G(I4_BODY)
VIOL_I4 = F(Exists("alpha", lambda a: CHECKED & ~grounded(a)))

def guard_c07(run, t, alpha):
    """g(s^a_t, alpha): block resume_line on a line the assistant has observed as expired."""
    return alpha.tool == "resume_line" and str(alpha.args.get("line_id", "")) in run.expired_lines(t)

FORBIDDEN = Atom("forbidden_g", lambda r, t: r.proposed(t) and guard_c07(r, t, r.occurring(t)[0]))
BLOCKED = Atom("blocked", lambda r, t: r.is_blocked(t))
EXECUTED_ANY = Atom("executed(any)", lambda r, t: r.proposed(t) and not r.is_blocked(t))
I5 = G(EXECUTED_ANY >> ~FORBIDDEN)                # executed => not forbidden
RESPONSE = G((ANYCALL & FORBIDDEN) >> BLOCKED)    # proposed & forbidden => blocked
SOUNDNESS = G(BLOCKED >> FORBIDDEN)
LACT_C07 = G(ANYCALL >> ~FORBIDDEN)               # the proposed-action version (not enforced)
