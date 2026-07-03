"""Tier-1 happens-before analyzer for TileLang-Ascend manual-CV kernels.

Builds a happens-before (HB) graph for each `T.Scope` directly from the Python
`ast` (design-doc "option 2": no tilelang/TVM/NPU install required) and reports
**cross-pipe data races** -- two accesses to the same on-chip buffer on
different hardware pipes, at least one a write, with no synchronizing path
between them. This is the precision that the Tier-0 balance lint structurally
cannot provide: Tier 0 proves set/wait counts match; Tier 1 proves the fences
actually order the conflicting memory accesses.

Model
-----
Each `T.Scope("C"|"V")` is one subcore. We linearize its body in source order
(recursing through loops/`if`s, flattening one pass) into events:

  * OP   : a data op (copy / mma / tile.* / reduce_*), with a hardware *pipe*
           and buffer read/write sets
  * SET  : T.set_flag(p, q, id)   -- issued on pipe p
  * WAIT : T.wait_flag(p, q, id)  -- issued on pipe q

Pipe of a `copy` is inferred from the *memory space* of its operands (GM / L1 /
L0A / L0B / L0C / UB), which we recover from the buffer declarations
(`alloc_L1` / `alloc_ub` / ... ) and the kernel's GM tensor parameters.

HB edges:
  * program order within each pipe (a pipe issues its events in order), and
  * each set_flag matched FIFO to the next wait_flag with the same (p, q, id):
    an edge SET -> WAIT (so everything on pipe p up to the set precedes
    everything on pipe q from the wait onward).

A race exists between conflicting accesses A (pipe a) and B (pipe b!=a) when
neither A reaches B nor B reaches A in this graph. This is intentionally a
single-pass (intra-iteration) analysis; cross-iteration hazards are out of scope
(see Limitations in the design doc).
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl_sync_lint import (  # noqa: E402
    ConstEvaluator,
    Diag,
    _attr_tail,
    collect_module_constants,
    find_auto_sync_disabled,
)


# --------------------------------------------------------------------------- #
# Memory spaces and pipe classification
# --------------------------------------------------------------------------- #
ALLOC_SPACE = {
    "alloc_shared": "L1",
    "alloc_L1": "L1",
    "alloc_L0A": "L0A",
    "alloc_L0B": "L0B",
    "alloc_L0C": "L0C",
    "alloc_fragment": "L0C",
    "alloc_ub": "UB",
}


def classify_copy_pipe(src: Optional[str], dst: Optional[str]) -> str:
    """Pipe that executes a copy from `src` space to `dst` space."""
    if src == "GM":
        return "MTE2"                       # GM -> L1/UB/L0 : load
    if dst == "GM":
        if src == "L0C":
            return "FIX"                    # L0C -> GM : cube fixpipe/write
        return "MTE3"                       # UB/L1 -> GM : store
    if src == "L1" and dst in ("L0A", "L0B"):
        return "MTE1"                       # L1 -> L0 : feed cube
    return "V"                              # UB<->UB and the rest : vector


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    idx: int
    line: int
    pipe: str
    kind: str                 # "op" | "set" | "wait"
    reads: list = field(default_factory=list)
    writes: list = field(default_factory=list)
    p: str = ""
    q: str = ""
    eid: object = None
    text: str = ""
    loopvars: tuple = ()      # enclosing for-loop target vars
    branch: tuple = ()        # path of (if-node-id, arm) guards
    subs: dict = field(default_factory=dict)  # buf -> (index_text, index_names)


# --------------------------------------------------------------------------- #
# Per-scope linearizer
# --------------------------------------------------------------------------- #
class ScopeLinearizer:
    def __init__(self, ev: ConstEvaluator, space: dict[str, str],
                 slot_renames: Optional[dict[str, str]] = None):
        self.ev = ev
        self.space = space
        self.slot_renames = slot_renames or {}
        self.events: list[Event] = []
        self._n = 0
        self._ctx_loopvars: tuple = ()
        self._ctx_branch: tuple = ()

    def run(self, body) -> list[Event]:
        for stmt in body:
            self._walk(stmt, (), ())
        return self.events

    def _walk(self, stmt, loopvars: tuple, branch: tuple):
        if isinstance(stmt, ast.For):
            tgt = stmt.target.id if isinstance(stmt.target, ast.Name) else None
            lv = loopvars + (tgt,) if tgt else loopvars
            for s in stmt.body:
                self._walk(s, lv, branch)
            for s in stmt.orelse:
                self._walk(s, lv, branch)
            return
        if isinstance(stmt, ast.If):
            bid = id(stmt)
            for s in stmt.body:
                self._walk(s, loopvars, branch + ((bid, 0),))
            for s in stmt.orelse:
                self._walk(s, loopvars, branch + ((bid, 1),))
            return
        if isinstance(stmt, ast.With):
            for s in stmt.body:
                self._walk(s, loopvars, branch)
            return
        self._ctx_loopvars = loopvars
        self._ctx_branch = branch
        for call in [n for n in ast.walk(stmt) if isinstance(n, ast.Call)]:
            self._emit(call)

    def _emit(self, call: ast.Call):
        name = _attr_tail(call.func)
        if name is None:
            return
        line = call.lineno
        if name == "set_flag" and len(call.args) >= 3:
            self._add(Event(self._next(), line, self._str(call.args[0]), "set",
                            p=self._str(call.args[0]), q=self._str(call.args[1]),
                            eid=self._eid(call.args[2])))
            return
        if name == "wait_flag" and len(call.args) >= 3:
            self._add(Event(self._next(), line, self._str(call.args[1]), "wait",
                            p=self._str(call.args[0]), q=self._str(call.args[1]),
                            eid=self._eid(call.args[2])))
            return
        if name in ("set_cross_flag", "wait_cross_flag"):
            return  # cross-core: not an intra-scope HB edge here
        if name == "barrier_all":
            self._add(Event(self._next(), line, "*", "barrier", text="barrier_all"))
            return
        if name == "copy" and len(call.args) >= 2:
            subs: dict = {}
            src = self._operand(call.args[0], subs)
            dst = self._operand(call.args[1], subs)
            pipe = classify_copy_pipe(self.space.get(src), self.space.get(dst))
            self._add(Event(self._next(), line, pipe, "op",
                            reads=[src] if src else [], writes=[dst] if dst else [],
                            text="copy", subs=subs))
            return
        if name == "mma" and len(call.args) >= 3:
            subs = {}
            a = self._operand(call.args[0], subs)
            b = self._operand(call.args[1], subs)
            c = self._operand(call.args[2], subs)
            reads = [x for x in (a, b) if x]
            init = self._kw_is_true(call, "init")
            if not init and c:
                reads.append(c)
            self._add(Event(self._next(), line, "M", "op",
                            reads=reads, writes=[c] if c else [], text="mma", subs=subs))
            return
        if name in ("reduce_max", "reduce_sum", "reduce") and len(call.args) >= 2:
            subs = {}
            src = self._operand(call.args[0], subs)
            dst = self._operand(call.args[1], subs)
            self._add(Event(self._next(), line, "V", "op",
                            reads=[src] if src else [], writes=[dst] if dst else [],
                            text=name, subs=subs))
            return
        # T.tile.<op>(dst, *srcs)  -> vector pipe
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Attribute) \
                and call.func.value.attr == "tile" and call.args:
            subs = {}
            dst = self._operand(call.args[0], subs)
            reads = [self._operand(a, subs) for a in call.args[1:]]
            reads = [r for r in reads if r]
            self._add(Event(self._next(), line, "V", "op",
                            reads=reads, writes=[dst] if dst else [],
                            text=f"tile.{name}", subs=subs))
            return

    # -- helpers ----------------------------------------------------------- #
    def _next(self) -> int:
        self._n += 1
        return self._n - 1

    def _add(self, e: Event):
        e.loopvars = self._ctx_loopvars
        e.branch = self._ctx_branch
        self.events.append(e)

    def _str(self, node) -> str:
        return str(node.value) if isinstance(node, ast.Constant) else ast.unparse(node)

    def _eid(self, node):
        v = self.ev.eval(node)
        if v is not None:
            return v
        # canonicalize double-buffer slot vars (s1/s2 = nr%2) so per-slot
        # fences on the same physical event match across pipeline stages.
        if self.slot_renames:
            node = _RenameNames(self.slot_renames).visit(
                ast.parse(ast.unparse(node), mode="eval").body)
        return ast.unparse(node)

    def _base(self, node) -> Optional[str]:
        """Root buffer name of a (possibly sliced/attr) expression."""
        while True:
            if isinstance(node, ast.Subscript):
                node = node.value
            elif isinstance(node, ast.Attribute):
                node = node.value
            else:
                break
        if isinstance(node, ast.Name) and node.id in self.space:
            return node.id
        return None

    def _index_info(self, node) -> tuple:
        """(index_text, index_names) over all subscripts of a buffer operand."""
        texts: list[str] = []
        names: set = set()
        cur = node
        while isinstance(cur, (ast.Subscript, ast.Attribute)):
            if isinstance(cur, ast.Subscript):
                texts.append(ast.unparse(cur.slice))
                names |= {n.id for n in ast.walk(cur.slice) if isinstance(n, ast.Name)}
            cur = cur.value
        text = "][".join(reversed(texts)) if texts else None
        return text, names

    def _operand(self, node, subs: dict) -> Optional[str]:
        """Base buffer name of an operand; also records its subscript in `subs`."""
        base = self._base(node)
        if base is None:
            return None
        if base not in subs:
            subs[base] = self._index_info(node)
        return base

    def _kw_is_true(self, call: ast.Call, key: str) -> bool:
        for kw in call.keywords:
            if kw.arg == key and isinstance(kw.value, ast.Constant):
                return kw.value.value is True
        return False


class _RenameNames(ast.NodeTransformer):
    def __init__(self, renames: dict[str, str]):
        self.renames = renames

    def visit_Name(self, node):
        if node.id in self.renames:
            return ast.copy_location(ast.Name(id=self.renames[node.id], ctx=node.ctx), node)
        return node


def build_slot_renames(tree: ast.Module, ev: ConstEvaluator) -> dict[str, str]:
    """Map double-buffer slot vars (`name = <expr> % k`, k a small constant) to a
    single canonical token per modulus, so per-slot fences/subscripts unify."""
    renames: dict[str, str] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.BinOp)
                and isinstance(node.value.op, ast.Mod)):
            k = ev.eval(node.value.right)
            if k is not None and 2 <= k <= 8:
                renames[node.targets[0].id] = f"__slot{k}"
    return renames


# --------------------------------------------------------------------------- #
# HB graph + reachability
# --------------------------------------------------------------------------- #
class HBGraph:
    def __init__(self, events: list[Event]):
        self.events = events
        self.adj: dict[int, list[int]] = {e.idx: [] for e in events}
        self._build()
        self._reach_cache: dict[int, set] = {}

    def _build(self):
        # program order within each pipe
        by_pipe: dict[str, list[Event]] = {}
        for e in self.events:
            by_pipe.setdefault(e.pipe, []).append(e)
        for evs in by_pipe.values():
            evs.sort(key=lambda e: e.idx)
            for a, b in zip(evs, evs[1:]):
                self.adj[a.idx].append(b.idx)
        # FIFO-matched set -> wait edges
        pending: dict[tuple, deque] = {}
        for e in self.events:
            if e.kind == "set":
                pending.setdefault((e.p, e.q, e.eid), deque()).append(e.idx)
            elif e.kind == "wait":
                q = pending.get((e.p, e.q, e.eid))
                if q:
                    self.adj[q.popleft()].append(e.idx)
        # barrier_all(): a full fence across every pipe
        for b in self.events:
            if b.kind == "barrier":
                for e in self.events:
                    if e.idx < b.idx:
                        self.adj[e.idx].append(b.idx)
                    elif e.idx > b.idx:
                        self.adj[b.idx].append(e.idx)

    def reaches(self, src: int, dst: int) -> bool:
        seen = self._reach_cache.get(src)
        if seen is None:
            seen = set()
            dq = deque([src])
            while dq:
                u = dq.popleft()
                for v in self.adj[u]:
                    if v not in seen:
                        seen.add(v)
                        dq.append(v)
            self._reach_cache[src] = seen
        return dst in seen


# Typical intra-core fence directions (producer pipe -> consumer pipe).
_FENCE_PAIRS: dict[tuple[str, str], tuple[str, str]] = {
    ("MTE2", "MTE1"): ("MTE2", "MTE1"),
    ("MTE2", "V"): ("MTE2", "V"),
    ("MTE1", "M"): ("MTE1", "M"),
    ("M", "MTE1"): ("M", "MTE1"),
    ("M", "FIX"): ("M", "FIX"),
    ("MTE1", "FIX"): ("MTE1", "FIX"),
    ("V", "MTE3"): ("V", "MTE3"),
}


def _mutually_exclusive(ba: tuple, bb: tuple) -> bool:
    """True if the two branch paths diverge into different arms of a common If,
    so the two statements never execute together within one loop iteration."""
    for ga, gb in zip(ba, bb):
        if ga[0] != gb[0]:
            return False          # diverged structure (both can run)
        if ga[1] != gb[1]:
            return True           # same If, different arm -> exclusive
    return False


def _provably_disjoint(ea: Event, eb: Event, buf: str) -> bool:
    """Sound suppression of a flagged pair when the two accesses can never touch
    the same element: identical subscript that varies with an enclosing loop var
    (distinct iterations -> distinct elements) AND mutually-exclusive branches
    (same iteration -> only one runs). Covers the `kv_ub[b_i]` if/else gather."""
    sa = ea.subs.get(buf)
    sb = eb.subs.get(buf)
    if not sa or not sb:
        return False
    ta, na = sa
    tb, nb = sb
    if ta is None or ta != tb:
        return False
    if not (na & set(ea.loopvars)) or not (nb & set(eb.loopvars)):
        return False
    return _mutually_exclusive(ea.branch, eb.branch)


def _race_producer_consumer(ea: Event, ma: str, eb: Event, mb: str):
    """Return (producer_pipe, consumer_pipe, insert_before_line)."""
    if ma == "w" and mb == "r":
        return ea.pipe, eb.pipe, eb.line
    if ma == "r" and mb == "w":
        return eb.pipe, ea.pipe, ea.line
    if ma == "w" and mb == "w":
        lo, hi = sorted((ea, eb), key=lambda e: e.line)
        return lo.pipe, hi.pipe, hi.line
    return ea.pipe, eb.pipe, eb.line


def _fence_fix(prod: str, cons: str, buf: str, before_line: int) -> str:
    p, q = _FENCE_PAIRS.get((prod, cons), (prod, cons))
    return (
        f"insert wait_flag({p!r}, {q!r}, <event>) before line {before_line} "
        f"so pipe {cons} does not read {buf!r} until pipe {prod} finishes "
        f"(then set_flag({p!r}, {q!r}, <event>) after the {prod} producer)"
    )


def detect_races(scope: str, events: list[Event]) -> list[Diag]:
    hb = HBGraph(events)
    # buffer -> list of (event, mode) where mode in {"r","w"}
    accesses: dict[str, list[tuple[Event, str]]] = {}
    for e in events:
        if e.kind != "op":
            continue
        for b in e.writes:
            accesses.setdefault(b, []).append((e, "w"))
        for b in e.reads:
            accesses.setdefault(b, []).append((e, "r"))

    diags: list[Diag] = []
    for buf, accs in sorted(accesses.items()):
        reported: set = set()
        for i in range(len(accs)):
            for j in range(i + 1, len(accs)):
                ea, ma = accs[i]
                eb, mb = accs[j]
                if ea.pipe == eb.pipe:
                    continue
                if ma == "r" and mb == "r":
                    continue
                if hb.reaches(ea.idx, eb.idx) or hb.reaches(eb.idx, ea.idx):
                    continue
                if _provably_disjoint(ea, eb, buf):
                    continue
                lo, hi = sorted((ea, eb), key=lambda e: e.line)
                key = (lo.pipe, hi.pipe)
                if key in reported:
                    continue
                reported.add(key)
                hazard = {("w", "w"): "WAW", ("r", "w"): "WAR/RAW",
                          ("w", "r"): "RAW"}[(ma, mb)]
                prod, cons, before = _race_producer_consumer(ea, ma, eb, mb)
                fix = _fence_fix(prod, cons, buf, before)
                diags.append(Diag(
                    "warning", "DATA_RACE",
                    f"scope {scope!r}: {hazard} on buffer {buf!r} — "
                    f"{lo.pipe}@L{lo.line} ({lo.text}) vs {hi.pipe}@L{hi.line} "
                    f"({hi.text}) with no fence between pipes. Fix: {fix}.",
                    line=lo.line))
    return diags


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _scope_name(withnode: ast.With) -> Optional[str]:
    for item in withnode.items:
        ce = item.context_expr
        if (isinstance(ce, ast.Call) and _attr_tail(ce.func) == "Scope"
                and ce.args and isinstance(ce.args[0], ast.Constant)):
            return str(ce.args[0].value)
    return None


def build_space_map(tree: ast.Module) -> dict[str, str]:
    """name -> memory space. GM from prim_func tensor params; on-chip from allocs."""
    space: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call):
            tail = _attr_tail(node.value.func)
            if tail in ALLOC_SPACE:
                space[node.targets[0].id] = ALLOC_SPACE[tail]
    # GM params: any FunctionDef arg annotated with a T.Tensor(...) call
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                ann = arg.annotation
                if isinstance(ann, ast.Call) and _attr_tail(ann.func) == "Tensor":
                    space.setdefault(arg.arg, "GM")
    return space


def analyze_source(source: str) -> list[Diag]:
    tree = ast.parse(source)
    # The HB race model assumes the programmer inserts all fences. When
    # TL_ASCEND_AUTO_SYNC is on, the compiler adds them and the source has none,
    # so race analysis is not applicable.
    if not find_auto_sync_disabled(tree):
        return []
    consts = collect_module_constants(tree)
    ev = ConstEvaluator(consts)
    space = build_space_map(tree)
    slot_renames = build_slot_renames(tree, ev)

    diags: list[Diag] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            scope = _scope_name(node)
            if scope is not None:
                events = ScopeLinearizer(ev, space, slot_renames).run(node.body)
                diags += detect_races(scope, events)
    diags.sort(key=lambda d: (d.line, d.code))
    return diags


def analyze_file(path: str) -> list[Diag]:
    with open(path, "r", encoding="utf-8") as f:
        return analyze_source(f.read())


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tier-1 TileLang HB race analyzer")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    total = 0
    for path in args.files:
        try:
            diags = analyze_file(path)
        except SyntaxError as e:
            print(f"{path}: error: [PARSE] {e}")
            total += 1
            continue
        if diags:
            for d in diags:
                print(d.format(path))
            total += len(diags)
        elif not args.quiet:
            print(f"{path}: OK (no data races found)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
