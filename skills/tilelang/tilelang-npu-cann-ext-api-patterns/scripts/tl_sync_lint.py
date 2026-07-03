"""Tier-0 static sync linter for TileLang-Ascend manual-CV kernels.

Parses a TileLang kernel `.py` file with the Python `ast` module (no tilelang,
TVM, or NPU runtime required) and checks the hand-written pipe/cross-core
synchronization for the bug classes seen in real kernels:

  * intra-core pipe-flag imbalance   -> AICore deadlock / pipe stall
      (a missing wait_flag, or an extra/duplicate set_flag)
  * a wait_cross_flag whose semaphore is never set anywhere -> deadlock
  * a whole scope with multi-pipe data ops but NO intra-core fences while
    TL_ASCEND_AUTO_SYNC is disabled -> data race

This is a *necessary-condition* checker (Tier 0): exact and high-confidence for
the flag-balance property, but it does not model program order or buffer-level
happens-before (that is Tier 1, over TIR). It runs on raw source on any machine.

Multiplicity model
-------------------
`set_flag(p,q,id)` / `wait_flag(p,q,id)` are matched per (scope, p, q, id); a
correct event has equal *symbolic* set and wait multiplicity. Multiplicity is a
small polynomial over two scales:

  * "1"   : an O(1) term  -> a statement outside the per-task loop
            (pre-loop bootstrap, post-loop drain)
  * "U"   : once per task  (U = tasks assigned to this core; symbolic)
  * "U*P" : once per loop iteration (P = sub-iterations per task, the period of
            a `g % P` index such as c_tile / tit; P is symbolic)

Guards map to coverage as follows (innermost relevant guard dominates):

  * `g < GT` / `g >= 1` (inequality on the loop var)        -> U*P  (full period)
  * `c_tile == k`        (equality on a periodic `g % P` var)-> U
  * `c_tile < K`         (inequality on a periodic var)      -> K·U  (K may be P-1)
  * `tit == num_steps-1` etc.                                -> U
  * a constant-trip nested loop `T.serial(NR)`               -> ×NR

Because `U*P` and `K·U` are expressed in the same basis, a software-pipeline
prologue/epilogue (`g<GT` vs `g>=1`), a resident reload/release pair
(`tit==0` / `tit==last`), and a resident buffer consumed every chunk but loaded
once per task all cancel exactly -- the legitimate asymmetries -- while a truly
unmatched set or wait does not.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
@dataclass
class Diag:
    severity: str  # "error" | "warning"
    code: str
    message: str
    line: int = 0

    def format(self, path: str) -> str:
        loc = f"{path}:{self.line}" if self.line else path
        return f"{loc}: {self.severity}: [{self.code}] {self.message}"


# --------------------------------------------------------------------------- #
# Constant evaluation
# --------------------------------------------------------------------------- #
class ConstEvaluator:
    """Best-effort folding of integer constants. Returns None when symbolic."""

    def __init__(self, consts: dict[str, Optional[int]]):
        self.consts = consts

    def eval(self, node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, int) else None
        if isinstance(node, ast.Name):
            return self.consts.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = self.eval(node.operand)
            return -v if v is not None else None
        if isinstance(node, ast.BinOp):
            l = self.eval(node.left)
            r = self.eval(node.right)
            if l is None or r is None:
                return None
            op = node.op
            try:
                if isinstance(op, ast.Add):
                    return l + r
                if isinstance(op, ast.Sub):
                    return l - r
                if isinstance(op, ast.Mult):
                    return l * r
                if isinstance(op, ast.FloorDiv):
                    return l // r
                if isinstance(op, ast.Mod):
                    return l % r
                if isinstance(op, ast.Div):
                    return l // r if l % r == 0 else None
            except ZeroDivisionError:
                return None
        return None


def collect_module_constants(tree: ast.Module) -> dict[str, Optional[int]]:
    """Three passes over top-level int assignments so forward refs resolve."""
    consts: dict[str, Optional[int]] = {}
    ev = ConstEvaluator(consts)
    for _ in range(3):
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name):
                    consts[tgt.id] = ev.eval(node.value)
                elif isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    if len(tgt.elts) == len(node.value.elts):
                        for nm, val in zip(tgt.elts, node.value.elts):
                            if isinstance(nm, ast.Name):
                                consts[nm.id] = ev.eval(val)
    return consts


def find_auto_sync_disabled(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is not None and _attr_tail(k) == "TL_ASCEND_AUTO_SYNC":
                    if isinstance(v, ast.Constant) and v.value is False:
                        return True
    return False


def _attr_tail(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


# --------------------------------------------------------------------------- #
# Multiplicity: a flat Counter over atoms {"1", "U", "U*P", "COND@<line>"}.
# --------------------------------------------------------------------------- #
def mult_is_zero(m: Counter) -> bool:
    return all(c == 0 for c in m.values())


def mult_str(m: Counter) -> str:
    order = {"1": 0, "U": 1, "U*P": 2}
    parts = []
    for sym in sorted(m, key=lambda s: (order.get(s, 9), s)):
        c = m[sym]
        if c:
            parts.append(f"{c:+d}*{sym}")
    return " ".join(parts) if parts else "0"


# --------------------------------------------------------------------------- #
# Context frames
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    kind: str                      # "loop" | "guard"
    var: str = ""
    const_trip: Optional[int] = None       # loop: integer trip, or None=symbolic
    gkind: str = ""                # guard: "periodic" | "loopvar" | "cond"
    cover: Counter = field(default_factory=Counter)  # guard coverage in U-basis
    line: int = 0


def frames_to_mult(frames: list[Frame]) -> Counter:
    const_loop = 1
    in_loop = False
    cover: Optional[Counter] = None
    cover_rank = -1  # cond=0 < loopvar=1 < periodic=2 ; innermost of equal rank wins
    cond_line = 0
    for f in frames:
        if f.kind == "loop":
            if f.const_trip is not None:
                const_loop *= f.const_trip
            else:
                in_loop = True
        elif f.kind == "guard":
            rank = {"cond": 0, "loopvar": 1, "periodic": 2}[f.gkind]
            if rank >= cover_rank:
                cover_rank = rank
                cover = Counter(f.cover)
                cond_line = f.line
    m: Counter = Counter()
    if not in_loop and cover is None:
        m["1"] += const_loop
        return m
    if cover_rank == 0:  # data-dependent guard -> opaque, won't cancel
        m[f"COND@{cond_line}"] += const_loop
        return m
    if cover is None:  # unguarded statement inside the symbolic loop
        m["U*P"] += const_loop
        return m
    for atom, c in cover.items():
        m[atom] += c * const_loop
    return m


# --------------------------------------------------------------------------- #
# Recorded operations
# --------------------------------------------------------------------------- #
@dataclass
class FlagOp:
    scope: str
    kind: str   # "set" | "wait"
    p: str
    q: str
    eid: object
    eid_text: str
    line: int
    mult: Counter


@dataclass
class CrossOp:
    scope: str
    kind: str   # "set" | "wait"
    sem: object
    sem_text: str
    line: int


@dataclass
class ScopeInfo:
    name: str
    n_intra_flag: int = 0
    n_copy: int = 0
    n_compute: int = 0
    line: int = 0


# --------------------------------------------------------------------------- #
# The walker
# --------------------------------------------------------------------------- #
class SyncWalker:
    def __init__(
        self,
        consts: dict[str, Optional[int]],
        intra_eid_names: dict[int, list[str]],
        cross_sem_names: dict[int, list[str]],
    ):
        self.ev = ConstEvaluator(consts)
        self.intra_eid_names = intra_eid_names
        self.cross_sem_names = cross_sem_names
        self.flag_ops: list[FlagOp] = []
        self.cross_ops: list[CrossOp] = []
        self.scopes: dict[str, ScopeInfo] = {}
        self.loop_vars: list[str] = []
        self.periodic_vars: dict[str, str] = {}   # name -> modulus expr string
        self.period_str: Optional[str] = None

    def run(self, tree: ast.Module):
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                scope = self._scope_name(node)
                if scope is not None:
                    info = ScopeInfo(name=scope, line=node.lineno)
                    self.scopes[scope] = info
                    self._walk_body(node.body, scope, [], info)

    @staticmethod
    def _scope_name(withnode: ast.With) -> Optional[str]:
        for item in withnode.items:
            ce = item.context_expr
            if (
                isinstance(ce, ast.Call)
                and _attr_tail(ce.func) == "Scope"
                and ce.args
                and isinstance(ce.args[0], ast.Constant)
            ):
                return str(ce.args[0].value)
        return None

    def _walk_body(self, body, scope, frames, info):
        for stmt in body:
            self._walk_stmt(stmt, scope, frames, info)

    def _walk_stmt(self, stmt, scope, frames, info):
        if isinstance(stmt, ast.For):
            frame = self._loop_frame(stmt)
            saved = (dict(self.periodic_vars), self.period_str)
            if frame.const_trip is None:
                self._register_periodic(stmt.body)
            self.loop_vars.append(frame.var)
            self._walk_body(stmt.body, scope, frames + [frame], info)
            self.loop_vars.pop()
            self.periodic_vars, self.period_str = saved
            self._walk_body(stmt.orelse, scope, frames, info)
            return

        if isinstance(stmt, ast.If):
            self._walk_body(stmt.body, scope, frames + [self._guard_frame(stmt.test)], info)
            if stmt.orelse:
                self._walk_body(
                    stmt.orelse, scope, frames + [self._guard_frame(stmt.test, negate=True)], info
                )
            return

        if isinstance(stmt, ast.With):
            if self._scope_name(stmt) is None:
                self._walk_body(stmt.body, scope, frames, info)
            return

        for call in self._calls_in(stmt):
            self._record_call(call, scope, frames, info)

    # -- periodic-variable detection --------------------------------------- #
    def _register_periodic(self, body):
        """Find `name = <expr> % <mod>` anywhere in the loop body (incl. nested
        under `if g>=1:` etc.). The loop period is the first *symbolic* modulus
        (e.g. tiles_c / num_steps / tpt), never a constant inner modulus."""
        sym_period: Optional[str] = None
        for stmt in body:
            for n in ast.walk(stmt):
                if (
                    isinstance(n, ast.Assign)
                    and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)
                    and isinstance(n.value, ast.BinOp)
                    and isinstance(n.value.op, ast.Mod)
                ):
                    mod = ast.unparse(n.value.right)
                    self.periodic_vars[n.targets[0].id] = mod
                    if self.ev.eval(n.value.right) is None and sym_period is None:
                        sym_period = mod
        if sym_period is not None:
            self.period_str = sym_period

    # -- frame construction ------------------------------------------------- #
    def _loop_frame(self, fornode: ast.For) -> Frame:
        var = fornode.target.id if isinstance(fornode.target, ast.Name) else "_"
        trip = None
        it = fornode.iter
        if isinstance(it, ast.Call):
            tail = _attr_tail(it.func)
            if tail in ("serial", "range", "Pipelined", "vectorized", "grid") and it.args:
                if tail == "range" and len(it.args) == 2:
                    a, b = self.ev.eval(it.args[0]), self.ev.eval(it.args[1])
                    trip = (b - a) if (a is not None and b is not None) else None
                else:
                    trip = self.ev.eval(it.args[0])
        return Frame(kind="loop", var=var, const_trip=trip, line=fornode.lineno)

    def _guard_frame(self, test: ast.expr, negate: bool = False) -> Frame:
        line = getattr(test, "lineno", 0)
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op = test.ops[0]
            left, right = test.left, test.comparators[0]
            if negate:
                op = _negate_cmp(op)
            # equality on a periodic var -> U
            if isinstance(op, ast.Eq):
                var = self._noncconst_side(left, right)
                if var in self.periodic_vars:
                    return Frame(kind="guard", gkind="periodic", var=var,
                                 cover=Counter({"U": 1}), line=line)
            if isinstance(op, ast.NotEq):
                # negated-eq handled above via _negate_cmp; bare != is data-dependent
                return Frame(kind="guard", gkind="cond", line=line)
            # inequality
            if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                lv = self._loopvar_in(left) or self._loopvar_in(right)
                if lv is not None:
                    return Frame(kind="guard", gkind="loopvar", var=lv,
                                 cover=Counter({"U*P": 1}), line=line)
                var, bound, flipped = self._ineq_parts(left, right)
                if var in self.periodic_vars:
                    cover = self._ineq_cover(op, bound, flipped)
                    if cover is not None:
                        return Frame(kind="guard", gkind="periodic", var=var,
                                     cover=cover, line=line)
        return Frame(kind="guard", gkind="cond", line=line)

    def _ineq_parts(self, left, right):
        """Return (var_str, bound_node, flipped) with var on the left logically."""
        lv = ast.unparse(left)
        rv = ast.unparse(right)
        if lv in self.periodic_vars:
            return lv, right, False
        if rv in self.periodic_vars:
            return rv, left, True
        return lv, right, False

    def _ineq_cover(self, op, bound_node, flipped) -> Optional[Counter]:
        poly = self._bound_poly(bound_node)
        if poly is None:
            return None
        # normalize so comparison reads `var <op> bound`
        if flipped:
            op = _flip_cmp(op)
        P = Counter({"U*P": 1})
        if isinstance(op, ast.Lt):       # var < bound  -> bound residues
            return poly
        if isinstance(op, ast.LtE):      # var <= bound -> bound+1
            return poly + Counter({"U": 1})
        if isinstance(op, ast.GtE):      # var >= bound -> P - bound
            return P - poly
        if isinstance(op, ast.Gt):       # var > bound  -> P - bound - 1
            return P - poly - Counter({"U": 1})
        return None

    def _bound_poly(self, node: ast.expr) -> Optional[Counter]:
        """Express a periodic-inequality bound as a*U + b*U*P (a=residue count)."""
        iv = self.ev.eval(node)
        if iv is not None:
            return Counter({"U": iv})
        if self.period_str is not None and ast.unparse(node) == self.period_str:
            return Counter({"U*P": 1})
        if isinstance(node, ast.BinOp) and self.period_str is not None:
            if ast.unparse(node.left) == self.period_str:
                k = self.ev.eval(node.right)
                if k is not None and isinstance(node.op, ast.Sub):
                    return Counter({"U*P": 1, "U": -k})
                if k is not None and isinstance(node.op, ast.Add):
                    return Counter({"U*P": 1, "U": k})
        return None

    def _noncconst_side(self, left, right) -> str:
        if isinstance(right, ast.Constant):
            return ast.unparse(left)
        if isinstance(left, ast.Constant):
            return ast.unparse(right)
        return ast.unparse(left)

    def _loopvar_in(self, node) -> Optional[str]:
        if isinstance(node, ast.Name) and node.id in self.loop_vars:
            return node.id
        return None

    # -- call extraction / recording --------------------------------------- #
    @staticmethod
    def _calls_in(stmt):
        return [n for n in ast.walk(stmt) if isinstance(n, ast.Call)]

    def _record_call(self, call, scope, frames, info):
        name = _attr_tail(call.func)
        if name is None:
            return
        line = call.lineno
        if name in ("set_flag", "wait_flag"):
            info.n_intra_flag += 1
            if len(call.args) >= 3:
                p = self._str_arg(call.args[0])
                q = self._str_arg(call.args[1])
                eid_text = ast.unparse(call.args[2])
                eid = self.ev.eval(call.args[2])
                if eid is None:
                    eid = eid_text
                self.flag_ops.append(FlagOp(
                    scope, "set" if name == "set_flag" else "wait",
                    p, q, eid, eid_text, line, frames_to_mult(frames)))
            return
        if name in ("set_cross_flag", "wait_cross_flag"):
            if name == "set_cross_flag" and len(call.args) >= 2:
                sem_text = ast.unparse(call.args[1])
                sem = self.ev.eval(call.args[1])
                self.cross_ops.append(CrossOp(
                    scope, "set",
                    sem if sem is not None else sem_text, sem_text, line))
            elif name == "wait_cross_flag" and len(call.args) >= 1:
                sem_text = ast.unparse(call.args[0])
                sem = self.ev.eval(call.args[0])
                self.cross_ops.append(CrossOp(
                    scope, "wait",
                    sem if sem is not None else sem_text, sem_text, line))
            return
        if name == "copy":
            info.n_copy += 1
        elif name in ("mma", "reduce_max", "reduce_sum", "reduce"):
            info.n_compute += 1
        elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Attribute):
            if call.func.value.attr == "tile":
                info.n_compute += 1

    @staticmethod
    def _str_arg(node) -> str:
        return str(node.value) if isinstance(node, ast.Constant) else ast.unparse(node)

    def _eid_label(self, eid, eid_text: str = "", *, cross: bool = False) -> str:
        if eid_text and _is_simple_eid(eid_text):
            return eid_text
        names = self.cross_sem_names if cross else self.intra_eid_names
        if isinstance(eid, int) and eid in names:
            ns = names[eid]
            if len(ns) == 1:
                return ns[0]
            return f"{ns[0]} (numeric id {eid}; also: {', '.join(ns[1:])})"
        return str(eid)


def _is_simple_eid(text: str) -> bool:
    """True for SIG_COMB_L1 or SIG_L0AB + 1, not for complex expressions."""
    return text.isidentifier() or (
        "+" in text and all(part.strip().isidentifier() or part.strip().isdigit()
                            for part in text.split("+"))
    )


def _pick_eid_label(ops: list[FlagOp], w: SyncWalker) -> str:
    for op in ops:
        if op.eid_text and _is_simple_eid(op.eid_text):
            return w._eid_label(op.eid, op.eid_text, cross=False)
    if ops:
        return w._eid_label(ops[0].eid, ops[0].eid_text, cross=False)
    return "?"


def _format_flag_imbalance(
    scope: str,
    p: str,
    q: str,
    eid_label: str,
    net: Counter,
    set_lines: list[int],
    wait_lines: list[int],
) -> str:
    surplus_set = any(c > 0 for c in net.values())
    set_s = sorted(set_lines)
    wait_s = sorted(wait_lines)
    if surplus_set:
        cause = (
            f"set_flag({p!r}, {q!r}, {eid_label}) occurs more often than "
            f"wait_flag({p!r}, {q!r}, {eid_label}) in scope {scope!r} "
            f"(net [{mult_str(net)}]). Pipe {p} will stall trying to set an "
            f"event that pipe {q} has not consumed."
        )
        fix = (
            f"add wait_flag({p!r}, {q!r}, {eid_label}) on pipe {q} before each "
            f"consumer that reads the data produced by pipe {p}"
        )
        if set_s and not wait_s:
            fix += f" (set only at line(s) {set_s}; no matching wait in this scope)"
        elif set_s:
            fix += f" (set at {set_s}, wait at {wait_s or 'none'})"
    else:
        cause = (
            f"wait_flag({p!r}, {q!r}, {eid_label}) occurs more often than "
            f"set_flag({p!r}, {q!r}, {eid_label}) in scope {scope!r} "
            f"(net [{mult_str(net)}]). Pipe {q} waits for data pipe {p} never signaled."
        )
        fix = (
            f"add set_flag({p!r}, {q!r}, {eid_label}) on pipe {p} after the "
            f"producer completes, before wait at line(s) {wait_s}"
        )
    return f"{cause} Fix: {fix}."


def _negate_cmp(op):
    return {ast.Eq: ast.NotEq(), ast.NotEq: ast.Eq(), ast.Lt: ast.GtE(),
            ast.LtE: ast.Gt(), ast.Gt: ast.LtE(), ast.GtE: ast.Lt()}.get(type(op), op)


def _flip_cmp(op):
    return {ast.Lt: ast.Gt(), ast.Gt: ast.Lt(), ast.LtE: ast.GtE(),
            ast.GtE: ast.LtE()}.get(type(op), op)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_intra_flags(w: SyncWalker) -> list[Diag]:
    diags: list[Diag] = []
    keys: dict[tuple, dict] = {}
    ops_by_key: dict[tuple, list[FlagOp]] = {}
    for op in w.flag_ops:
        k = (op.scope, op.p, op.q, op.eid)
        entry = keys.setdefault(k, {"net": Counter(), "set": [], "wait": []})
        ops_by_key.setdefault(k, []).append(op)
        if op.kind == "set":
            entry["net"].update(op.mult)
            entry["set"].append(op.line)
        else:
            entry["net"].subtract(op.mult)
            entry["wait"].append(op.line)
    for key, entry in sorted(keys.items(), key=lambda kv: str(kv[0])):
        scope, p, q, eid = key
        net = entry["net"]
        if not mult_is_zero(net):
            eid_label = _pick_eid_label(ops_by_key[key], w)
            first = min(entry["set"] + entry["wait"])
            diags.append(Diag(
                "error", "FLAG_IMBALANCE",
                _format_flag_imbalance(
                    scope, p, q, eid_label, net,
                    entry["set"], entry["wait"]),
                line=first))
    return diags


def check_cross_flags(w: SyncWalker) -> list[Diag]:
    diags: list[Diag] = []
    sems: dict[object, dict] = {}
    for op in w.cross_ops:
        e = sems.setdefault(op.sem, {"set": [], "wait": [], "text": ""})
        e[op.kind].append(op.line)
        if op.sem_text and not e["text"]:
            e["text"] = op.sem_text
    for sem, e in sorted(sems.items(), key=lambda kv: str(kv[0])):
        if e["wait"] and not e["set"]:
            sem_text = e.get("text", "")
            sem_label = w._eid_label(sem, sem_text, cross=True)
            diags.append(Diag(
                "error", "CROSS_DEADLOCK",
                f"scope waits on cross-core semaphore {sem_label} "
                f"(wait_cross_flag at line(s) {sorted(e['wait'])}) but it is never "
                f"set_cross_flag anywhere in the kernel -> the waiter blocks forever "
                f"(AICore timeout 507014). Fix: add set_cross_flag(..., {sem_label}) "
                f"in the producer scope before the matching wait.",
                line=min(e["wait"])))
    return diags


def check_missing_fences(w: SyncWalker, auto_sync_off: bool) -> list[Diag]:
    diags: list[Diag] = []
    if not auto_sync_off:
        return diags
    for name, info in sorted(w.scopes.items()):
        if info.n_intra_flag == 0 and info.n_copy > 0 and info.n_compute > 0:
            diags.append(Diag(
                "warning", "NO_INTRA_FENCE",
                f"scope {name!r} has data-movement copies ({info.n_copy}) and compute "
                f"ops ({info.n_compute}) on multiple pipes but NO intra-core "
                f"wait_flag/set_flag, while TL_ASCEND_AUTO_SYNC is disabled -> cross-pipe "
                f"ordering (MTE2 load -> V compute -> MTE3 store) is unsynchronized "
                f"(data race). flash_attn_opt.py fences its vector scope explicitly.",
                line=info.line))
    return diags


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def lint_source(source: str) -> list[Diag]:
    tree = ast.parse(source)
    consts = collect_module_constants(tree)
    auto_sync_off = find_auto_sync_disabled(tree)

    intra_eid_names: dict[int, list[str]] = {}
    cross_sem_names: dict[int, list[str]] = {}
    for nm, val in consts.items():
        if isinstance(val, int):
            if nm.startswith("SIG_"):
                intra_eid_names.setdefault(val, []).append(nm)
            elif nm.startswith("SEM_"):
                cross_sem_names.setdefault(val, []).append(nm)

    w = SyncWalker(consts, intra_eid_names, cross_sem_names)
    w.run(tree)

    diags = (
        check_intra_flags(w)
        + check_cross_flags(w)
        + check_missing_fences(w, auto_sync_off)
    )
    diags.sort(key=lambda d: (d.line, d.code))
    return diags


def lint_file(path: str) -> list[Diag]:
    with open(path, "r", encoding="utf-8") as f:
        return lint_source(f.read())


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tier-0 TileLang sync linter")
    parser.add_argument("files", nargs="+", help="kernel .py file(s) to lint")
    parser.add_argument("--quiet", action="store_true", help="only print on findings")
    parser.add_argument("--tier1", action="store_true",
                        help="also run the Tier-1 happens-before data-race analyzer")
    parser.add_argument("--tier2", action="store_true",
                        help="also run the Tier-2 deferred-credit (late-release) "
                             "deadlock check")
    parser.add_argument("--tier3", action="store_true",
                        help="also run the Tier-3 first-iteration credit-underflow "
                             "(prime-balance) deadlock check")
    parser.add_argument("--tier4", action="store_true",
                        help="also run the Tier-4 cross-vid alloc_shared memory "
                             "violation check (CROSS_VID_SHARED)")
    args = parser.parse_args(argv)

    tier1 = None
    if args.tier1:
        import tl_sync_hb as tier1  # noqa: F401
    tier2 = None
    if args.tier2:
        import tl_sync_cycle as tier2  # noqa: F401
    tier3 = None
    if args.tier3:
        import tl_sync_prime as tier3  # noqa: F401
    tier4 = None
    if args.tier4:
        import tl_sync_mem as tier4  # noqa: F401

    total_errors = 0
    for path in args.files:
        try:
            diags = lint_file(path)
            if tier1 is not None:
                diags = diags + tier1.analyze_file(path)
            if tier2 is not None:
                diags = diags + tier2.analyze_file(path)
            if tier3 is not None:
                diags = diags + tier3.analyze_file(path)
            if tier4 is not None:
                diags = diags + tier4.analyze_file(path)
            if tier1 is not None or tier2 is not None or tier3 is not None or tier4 is not None:
                diags.sort(key=lambda d: (d.line, d.code))
        except SyntaxError as e:
            print(f"{path}: error: [PARSE] {e}")
            total_errors += 1
            continue
        total_errors += sum(1 for d in diags if d.severity == "error")
        if diags:
            for d in diags:
                print(d.format(path))
        elif not args.quiet:
            print(f"{path}: OK (no sync issues found)")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
