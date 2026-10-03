"""Canonical trace adapter for tau2-bench released simulations.

Turns a simulation record into an ordered list of steps with a uniform schema,
so that the coverage, grounding and warning analyses all read the same object.
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    idx: int                 # position in the canonical step sequence
    turn_idx: int
    actor: str               # 'assistant' | 'user'
    kind: str                # 'message' | 'tool_call' | 'tool_result'
    text: str = ""
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    error: bool = False
    call_id: str | None = None

    def serialize(self) -> str:
        if self.kind == "tool_call":
            return f"{self.actor}:CALL:{self.tool}:" + " ".join(
                f"{k}={v}" for k, v in sorted(self.args.items())
            )
        if self.kind == "tool_result":
            return f"{self.actor}:RESULT:{self.tool}:{'ERR:' if self.error else ''}{self.text[:400]}"
        return f"{self.actor}:MSG:{self.text[:400]}"


@dataclass
class Trace:
    sim_id: str
    task_id: str
    trial: int
    reward: float
    termination_reason: str
    duration: float
    agent_cost: float | None
    steps: list[Step]
    task: dict[str, Any] | None = None
    # per-assistant-message tool-call counts, and messages mixing prose with a call
    msg_call_counts: list[int] = field(default_factory=list)
    mixed_msgs: int = 0

    # --- task metadata parsed from the task id -------------------------------
    @property
    def issue_type(self) -> str:
        m = re.match(r"\[([^\]]+)\]", self.task_id)
        return m.group(1) if m else ""

    @property
    def faults(self) -> list[str]:
        body = re.sub(r"^\[[^\]]+\]", "", self.task_id)
        body = re.sub(r"\[PERSONA:[^\]]*\]$", "", body)
        return [f for f in body.split("|") if f]

    @property
    def failed(self) -> bool:
        return self.reward < 1.0

    # --- convenience views ---------------------------------------------------
    def calls(self, actor: str | None = None) -> list[Step]:
        return [s for s in self.steps
                if s.kind == "tool_call" and (actor is None or s.actor == actor)]

    def call_names(self, actor: str | None = None) -> list[str]:
        return [s.tool for s in self.calls(actor)]

    def result_for(self, call: Step) -> Step | None:
        for s in self.steps[call.idx + 1:]:
            if s.kind == "tool_result" and s.call_id == call.call_id:
                return s
        return None

    def first_index_of(self, tools: set[str], actor: str | None = None) -> int | None:
        for s in self.calls(actor):
            if s.tool in tools:
                return s.idx
        return None

    def assistant_messages(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "message" and s.actor == "assistant"]


def _result_text(msg: dict) -> str:
    c = msg.get("content")
    if c is None:
        return ""
    return c if isinstance(c, str) else json.dumps(c)


def build_trace(sim: dict, task: dict | None = None) -> Trace:
    steps: list[Step] = []
    idx = 0
    for m in sim["messages"]:
        role = m.get("role")
        turn = m.get("turn_idx", -1)
        content = _result_text(m)
        if role == "tool":
            # requestor tells us whose tool call this answers
            steps.append(Step(idx, turn, m.get("requestor", "assistant"), "tool_result",
                              text=content, error=bool(m.get("error")),
                              call_id=m.get("id")))
            idx += 1
            continue
        if content.strip():
            steps.append(Step(idx, turn, role, "message", text=content))
            idx += 1
        for tc in (m.get("tool_calls") or []):
            steps.append(Step(idx, turn, tc.get("requestor", role), "tool_call",
                              tool=tc.get("name"), args=tc.get("arguments") or {},
                              call_id=tc.get("id")))
            idx += 1
    # attach tool names to results by matching call ids
    by_id = {s.call_id: s for s in steps if s.kind == "tool_call"}
    for s in steps:
        if s.kind == "tool_result" and s.call_id in by_id:
            s.tool = by_id[s.call_id].tool
    msg_call_counts, mixed = [], 0
    for m in sim["messages"]:
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if tcs:
            msg_call_counts.append(len(tcs))
            if _result_text(m).strip():
                mixed += 1
    ri = sim.get("reward_info") or {}
    return Trace(
        sim_id=sim["id"], task_id=sim["task_id"], trial=sim.get("trial", 0),
        reward=float(ri.get("reward", 0.0)),
        termination_reason=sim.get("termination_reason", ""),
        duration=float(sim.get("duration") or 0.0),
        agent_cost=sim.get("agent_cost"),
        steps=steps, task=task,
        msg_call_counts=msg_call_counts, mixed_msgs=mixed,
    )


def load(path: str) -> list[Trace]:
    with open(path) as fh:
        d = json.load(fh)
    tasks = {t["id"]: t for t in d.get("tasks", [])}
    return [build_trace(s, tasks.get(s["task_id"])) for s in d["simulations"]]


# messages an assistant turn may hold alongside a tool call, per raw record
def raw_messages(path: str):
    with open(path) as fh:
        return json.load(fh)


if __name__ == "__main__":
    import sys, collections
    traces = load(sys.argv[1])
    print(f"{len(traces)} traces, {len(set(t.task_id for t in traces))} tasks")
    print("reward:", collections.Counter(t.reward for t in traces))
    print("issues:", collections.Counter(t.issue_type for t in traces))
    print("steps: mean %.1f  max %d" % (
        sum(len(t.steps) for t in traces) / len(traces),
        max(len(t.steps) for t in traces)))
    t = traces[0]
    print("example faults:", t.faults, "| calls:", t.call_names()[:8])
