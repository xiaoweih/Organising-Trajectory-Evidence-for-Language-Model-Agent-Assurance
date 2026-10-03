"""A reference checker for the logic of runs (Section 3 of the paper).

Run formulas are evaluated at positions (run, t).  A run is any object exposing
`positions()` (number of positions) and an `atom(name, t, *args)` method; the
`Run` class in `model.py` builds one from a tau2-bench trace.

Syntax (Definition 3.11):
    p | ~phi | phi & phi | X phi | phi U phi | Y phi | phi S phi
    | Forall(alpha) phi | Kobs phi | Ka phi | <arg(alpha)> chi
Argument formulas:  standing | ~chi | chi & chi | Gdep chi

Kobs / Ka are not evaluated by enumeration (Runs(Sigma) is not enumerable);
`Kobs` and `Ka` nodes therefore carry a *determinacy witness*: a function saying
whether the sub-formula's value at t is a function of the log prefix / the
local state.  This is enough to decide fragment membership syntactically for the
formulas the paper uses (Section 3.5), which is the only use the experiments
make of the epistemic operators.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any, Iterable


class Formula:
    def __and__(self, o): return And(self, o)
    def __or__(self, o): return Or(self, o)
    def __invert__(self): return Not(self)
    def __rshift__(self, o): return Implies(self, o)   # a >> b  ==  a -> b


@dataclass(frozen=True)
class Atom(Formula):
    name: str
    fn: Callable[[Any, int], bool]          # (run, t) -> bool
    forward: bool = False                   # value depends on positions after t
    local: bool = True                      # determined by the assistant's local state
    def __repr__(self): return self.name


@dataclass(frozen=True)
class Not(Formula):
    a: Formula
    def __repr__(self): return f"~{self.a}"

@dataclass(frozen=True)
class And(Formula):
    a: Formula; b: Formula
    def __repr__(self): return f"({self.a} & {self.b})"

@dataclass(frozen=True)
class Or(Formula):
    a: Formula; b: Formula
    def __repr__(self): return f"({self.a} | {self.b})"

@dataclass(frozen=True)
class Implies(Formula):
    a: Formula; b: Formula
    def __repr__(self): return f"({self.a} -> {self.b})"

@dataclass(frozen=True)
class X(Formula):
    a: Formula
    def __repr__(self): return f"X {self.a}"

@dataclass(frozen=True)
class U(Formula):
    a: Formula; b: Formula
    def __repr__(self): return f"({self.a} U {self.b})"

@dataclass(frozen=True)
class Y(Formula):
    a: Formula
    def __repr__(self): return f"Y {self.a}"

@dataclass(frozen=True)
class S(Formula):
    a: Formula; b: Formula
    def __repr__(self): return f"({self.a} S {self.b})"

@dataclass(frozen=True)
class Fle(Formula):          # F_{<= d}
    a: Formula; d: int
    def __repr__(self): return f"F<={self.d} {self.a}"

@dataclass(frozen=True)
class Oge(Formula):          # O_{>= l}
    a: Formula; l: int
    def __repr__(self): return f"O>={self.l} {self.a}"

@dataclass(frozen=True)
class Forall(Formula):
    """Quantification over the action instances occurring at the position
    (or over a finite data domain supplied by `domain(run, t)`)."""
    var: str
    body: Callable[[Any], Formula]         # alpha -> formula
    domain: Callable[[Any, int], Iterable[Any]] | None = None
    def __repr__(self): return f"forall {self.var}. ..."

@dataclass(frozen=True)
class Exists(Formula):
    var: str
    body: Callable[[Any], Formula]
    domain: Callable[[Any, int], Iterable[Any]] | None = None
    def __repr__(self): return f"exists {self.var}. ..."

@dataclass(frozen=True)
class Kobs(Formula):
    a: Formula
    def __repr__(self): return f"Kobs {self.a}"

@dataclass(frozen=True)
class Ka(Formula):
    a: Formula
    def __repr__(self): return f"Ka {self.a}"

@dataclass(frozen=True)
class ArgBridge(Formula):    # <arg(alpha)> chi
    alpha: Any
    chi: "ArgFormula"
    def __repr__(self): return f"<arg({self.alpha})> {self.chi}"


# ---- derived operators -----------------------------------------------------
TRUE = Atom("true", lambda r, t: True, local=True)
FALSE = Not(TRUE)
def F(a): return U(TRUE, a)
def G(a): return Not(F(Not(a)))
def O(a): return S(TRUE, a)
def H(a): return Not(O(Not(a)))
END = Atom("end", lambda r, t: t == r.positions() - 1)


# ---- argument formulas -----------------------------------------------------
class ArgFormula: pass

@dataclass(frozen=True)
class Standing(ArgFormula):
    def __repr__(self): return "standing"

@dataclass(frozen=True)
class ANot(ArgFormula):
    a: ArgFormula

@dataclass(frozen=True)
class AAnd(ArgFormula):
    a: ArgFormula; b: ArgFormula

@dataclass(frozen=True)
class Gdep(ArgFormula):
    a: ArgFormula
    def __repr__(self): return f"Gdep {self.a}"

STANDING = Standing()
GROUNDED_CHI = Gdep(STANDING)


@dataclass
class ArgStructure:
    """D = (V, R_dep, St) with arg: action instance -> node (partial)."""
    V: set = field(default_factory=set)
    dep: dict = field(default_factory=dict)       # x -> set of y with x R_dep y
    St: set = field(default_factory=set)
    arg: dict = field(default_factory=dict)       # alpha -> node

    def reach(self, x):
        seen, stack = set(), [x]
        while stack:
            v = stack.pop()
            if v in seen: continue
            seen.add(v); stack.extend(self.dep.get(v, ()))
        return seen

    def sat(self, x, chi) -> bool:
        if isinstance(chi, Standing): return x in self.St
        if isinstance(chi, ANot): return not self.sat(x, chi.a)
        if isinstance(chi, AAnd): return self.sat(x, chi.a) and self.sat(x, chi.b)
        if isinstance(chi, Gdep): return all(self.sat(y, chi.a) for y in self.reach(x))
        raise TypeError(chi)


# ---- evaluation -------------------------------------------------------------
class Checker:
    def __init__(self, run):
        self.run = run
        self.n = run.positions()
        self.memo: dict = {}

    def ev(self, f: Formula, t: int) -> bool:
        key = (f, t)
        if key in self.memo: return self.memo[key]
        v = self._ev(f, t)
        self.memo[key] = v
        return v

    def _ev(self, f, t):
        r, n = self.run, self.n
        if isinstance(f, Atom): return bool(f.fn(r, t))
        if isinstance(f, Not): return not self.ev(f.a, t)
        if isinstance(f, And): return self.ev(f.a, t) and self.ev(f.b, t)
        if isinstance(f, Or): return self.ev(f.a, t) or self.ev(f.b, t)
        if isinstance(f, Implies): return (not self.ev(f.a, t)) or self.ev(f.b, t)
        if isinstance(f, X): return t + 1 < n and self.ev(f.a, t + 1)
        if isinstance(f, Y): return t > 0 and self.ev(f.a, t - 1)
        if isinstance(f, U):
            for j in range(t, n):
                if self.ev(f.b, j): return True
                if not self.ev(f.a, j): return False
            return False
        if isinstance(f, S):
            for j in range(t, -1, -1):
                if self.ev(f.b, j): return True
                if not self.ev(f.a, j): return False
            return False
        if isinstance(f, Fle):
            return any(self.ev(f.a, j) for j in range(t, min(n, t + f.d + 1)))
        if isinstance(f, Oge):
            return any(self.ev(f.a, j) for j in range(0, t - f.l + 1))
        if isinstance(f, (Forall, Exists)):
            dom = f.domain(r, t) if f.domain else r.occurring(t)
            vals = [self.ev(f.body(a), t) for a in dom]
            return all(vals) if isinstance(f, Forall) else any(vals)
        if isinstance(f, (Kobs, Ka)):
            # evaluated as the sub-formula; determinacy is checked separately
            return self.ev(f.a, t)
        if isinstance(f, ArgBridge):
            D = r.argstruct(t)
            node = D.arg.get(f.alpha) if D else None
            return node is not None and D.sat(node, f.chi)
        raise TypeError(f)

    def holds(self, f) -> bool:
        return self.ev(f, 0)


# ---- syntactic fragment analysis (Section 3.5) ------------------------------
def is_forward(f) -> bool:
    """Does the value of f at t depend on positions after t?"""
    if isinstance(f, Atom): return f.forward
    if isinstance(f, (X, U, Fle)): return True
    if isinstance(f, (Not, Y, S, Oge, Kobs, Ka)): return is_forward(getattr(f, 'a'))  or (isinstance(f, S) and is_forward(f.b))
    if isinstance(f, (And, Or, Implies)): return is_forward(f.a) or is_forward(f.b)
    if isinstance(f, (Forall, Exists)):
        probe = f.body("__probe__")
        return is_forward(probe)
    if isinstance(f, ArgBridge): return False
    raise TypeError(f)

def is_local(f) -> bool:
    """Is f determined by the assistant's local state at t (and the actions it proposes)?"""
    if isinstance(f, Atom): return f.local
    if isinstance(f, (Not, X, Y, Fle, Oge, Kobs, Ka)): return is_local(f.a)
    if isinstance(f, (And, Or, Implies, U, S)): return is_local(f.a) and is_local(f.b)
    if isinstance(f, (Forall, Exists)): return is_local(f.body("__probe__"))
    if isinstance(f, ArgBridge): return True
    raise TypeError(f)

def uses_argument_formula(f) -> bool:
    if isinstance(f, ArgBridge): return True
    if isinstance(f, Atom): return False
    if isinstance(f, (Forall, Exists)): return uses_argument_formula(f.body("__probe__"))
    return any(uses_argument_formula(getattr(f, k)) for k in ('a', 'b') if hasattr(f, k))
