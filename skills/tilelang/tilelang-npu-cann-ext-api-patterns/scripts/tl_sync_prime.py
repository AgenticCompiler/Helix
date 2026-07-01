"""Tier-3 first-iteration credit-underflow ("prime-balance") check.

Catches a deadlock class that Tiers 0-2 miss: a `wait_flag` that is reachable on
the *first executed iteration* of a scope's main loop with **zero** possible
tokens, because the credit it consumes is neither primed in the prologue nor
produced by a guaranteed earlier producer in that first iteration. The only
producers are guarded by a phase condition that need not hold at loop entry, or
are loop-carried (they feed the *next* iteration). A semaphore cannot go
negative, so such a wait is an unconditional structural hang.

Motivating ground truth: `DLBlas/tilelang/indexer/indexer_npu_opt.py` hangs at
the vector scope's `wait_flag("MTE2","V",SIG_W)`. The SIG_W credit in that
direction is produced only under `if t_tile == 0:` (a fresh weight load) or by a
loop-carried re-grant; the prologue primes only the reverse direction. For cores
whose task range starts mid-group (`my_start % num_t_tiles != 0`) `t_tile != 0`
on iteration 1, the guarded producer is skipped, and the wait blocks forever.

Model (see tl_sync_prime_design.md)
-----------------------------------
Per scope, reuse the Tier-1 linearization (pipes + flag events + branch path):

1. Split into prologue (priming `set`s before the main loop) and loop body.
2. Walk prologue then ONE first-executed iteration in program order, keeping a
   token count per credit `(p, q, id)`. Prologue/earlier same-iteration `set`s
   add tokens; a `wait` consumes one and must find >= 1.
3. **Adversarial guard havoc.** A guard whose truth value cannot be decided at
   loop entry is treated nondeterministically, worst-case *per op kind*:
     * a guarded `set` (producer) is assumed **skipped** (contributes no token);
     * loop-var inequality guards (`g >= 1`, `g < gt`) are satisfied on the first
       executed iteration (the body runs), so they are honored as TRUE;
     * periodic-phase guards (`t_tile == 0`, on a `var = expr % mod`) and other
       data-dependent guards are undecidable -> havoc.
4. A `wait` that **provably runs** on the first iteration (unguarded or under a
   provably-true guard) with 0 tokens is reported as `PRIME_UNDERFLOW` (error).
   Guarded/havoc waits are not reported (precision choice) and do not decrement.

Loop-carried producers are handled for free: a `set` that appears *after* the
wait in program order has not been counted yet when the wait is checked.

We accept some false positives under the adversarial default (a correct kernel
that relies on a guaranteed entry guard to cover an unprimed wait will be
flagged); a later refutation layer (constant/phase reasoning, then SMT over the
task-partition constraints) can discharge them. Needs no tilelang / TVM / NPU.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl_sync_lint import (  # noqa: E402
    ConstEvaluator,
    Diag,
    _attr_tail,
    collect_module_constants,
    find_auto_sync_disabled,
)
from tl_sync_hb import (  # noqa: E402
    ScopeLinearizer,
    _scope_name,
    build_slot_renames,
    build_space_map,
)

_LOOP_FUNCS = ("serial", "Pipelined", "range", "grid", "vectorized")


def _split_loop(withbody) -> tuple[list, Optional[ast.For]]:
    """Split a scope body into (prologue stmts, main loop)."""
    pro: list = []
    for st in withbody:
        if (isinstance(st, ast.For) and isinstance(st.iter, ast.Call)
                and _attr_tail(st.iter.func) in _LOOP_FUNCS):
            return pro, st
        pro.append(st)
    return pro, None


# A double-buffer credit `SIG_BASE + s` (slot var canonicalized to __slotK by
# Tier 1) or `SIG_BASE + <int>` (literal slot, e.g. prologue priming both slots)
# is the same physical reuse credit; key tokens by its base symbol so the
# prologue's per-slot priming covers the body's per-slot wait.
_SLOT_RE = re.compile(r"^(.*?)\s*\+\s*(?:__slot\d+|\d+)$")


def _canon_eid(eid):
    if isinstance(eid, str):
        m = _SLOT_RE.match(eid)
        if m:
            return m.group(1).strip()
    return eid


def _names(node: ast.AST) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _collect_periodic(loop_body) -> set:
    """Names assigned `name = <expr> % <mod>` anywhere in the loop body."""
    periodic: set = set()
    for stmt in loop_body:
        for n in ast.walk(stmt):
            if (isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)
                    and isinstance(n.value, ast.BinOp)
                    and isinstance(n.value.op, ast.Mod)):
                periodic.add(n.targets[0].id)
    return periodic


def _first_exec_value(loop_body, loopvar: str, ev: ConstEvaluator) -> int:
    """First loop value at which the body actually runs. If the body is wrapped
    in a lower-bound guard on the loop var (`if g >= k` / `if g > k`), that k (or
    k+1) is the first executing index; otherwise 0."""
    for st in loop_body:
        if isinstance(st, ast.If) and isinstance(st.test, ast.Compare) \
                and len(st.test.ops) == 1:
            op = st.test.ops[0]
            left, right = st.test.left, st.test.comparators[0]
            if isinstance(left, ast.Name) and left.id == loopvar:
                k = ev.eval(right)
                if k is not None and isinstance(op, ast.GtE):
                    return k
                if k is not None and isinstance(op, ast.Gt):
                    return k + 1
    return 0


def _cmp(l: int, op: ast.cmpop, r: int) -> Optional[bool]:
    if isinstance(op, ast.Eq):
        return l == r
    if isinstance(op, ast.NotEq):
        return l != r
    if isinstance(op, ast.Lt):
        return l < r
    if isinstance(op, ast.LtE):
        return l <= r
    if isinstance(op, ast.Gt):
        return l > r
    if isinstance(op, ast.GtE):
        return l >= r
    return None


class GuardClassifier:
    """Decide the truth value of a guard on the first executed iteration:
    True / False / None(=undecidable -> havoc)."""

    def __init__(self, loopvar: str, periodic: set, ev: ConstEvaluator,
                 ev_bound: ConstEvaluator):
        self.loopvar = loopvar
        self.periodic = periodic
        self.ev = ev
        self.ev_bound = ev_bound

    def truth(self, test: ast.expr) -> Optional[bool]:
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
            return None
        op = test.ops[0]
        left, right = test.left, test.comparators[0]
        involves = self.loopvar in (_names(left) | _names(right))
        # Loop-var inequality: satisfied on the first executed iteration.
        if involves and isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            l, r = self.ev_bound.eval(left), self.ev_bound.eval(right)
            if l is not None and r is not None:
                return _cmp(l, op, r)
            return True
        if involves and isinstance(op, (ast.Eq, ast.NotEq)):
            l, r = self.ev_bound.eval(left), self.ev_bound.eval(right)
            if l is not None and r is not None:
                return _cmp(l, op, r)
            return None
        # Periodic-phase guard -> undecidable at entry.
        var = (ast.unparse(left) if not isinstance(left, ast.Constant)
               else ast.unparse(right))
        if var in self.periodic:
            return None
        # Constant-foldable guard.
        l, r = self.ev.eval(left), self.ev.eval(right)
        if l is not None and r is not None:
            return _cmp(l, op, r)
        return None

    def path_status(self, branch: tuple, bid_test: dict) -> str:
        """'run' (provably executes), 'skip' (provably not), or 'havoc'."""
        status = "run"
        for bid, arm in branch:
            test = bid_test.get(bid)
            if test is None:
                status = "havoc"
                continue
            v = self.truth(test)
            taken = v if arm == 0 else (None if v is None else not v)
            if taken is False:
                return "skip"
            if taken is None:
                status = "havoc"
        return status


def _eid_label(eid, sig_names: dict) -> str:
    if isinstance(eid, int) and eid in sig_names:
        return sig_names[eid]
    return str(eid)


def detect_prime_underflow(scope: str, pro, body, loop: ast.For,
                           consts: dict, ev: ConstEvaluator,
                           sig_names: dict) -> list[Diag]:
    loopvar = loop.target.id if isinstance(loop.target, ast.Name) else "_"
    periodic = _collect_periodic(loop.body)
    first_exec = _first_exec_value(loop.body, loopvar, ev)
    ev_bound = ConstEvaluator({**consts, loopvar: first_exec})
    gc = GuardClassifier(loopvar, periodic, ev, ev_bound)

    # bid -> If.test for every If in the loop body (ids match the linearizer's).
    bid_test: dict = {}
    for st in loop.body:
        for n in ast.walk(st):
            if isinstance(n, ast.If):
                bid_test[id(n)] = n.test

    tokens: dict[tuple, int] = {}
    # Prologue priming: only unguarded sets (prologue has no phase guards).
    for e in pro:
        if e.kind == "set":
            k = (e.p, e.q, _canon_eid(e.eid))
            tokens[k] = tokens.get(k, 0) + 1

    diags: list[Diag] = []
    reported: set = set()
    for e in body:
        if e.kind not in ("set", "wait"):
            continue
        key = (e.p, e.q, _canon_eid(e.eid))
        status = gc.path_status(e.branch, bid_test)
        if status == "skip":
            continue
        if e.kind == "set":
            if status == "run":            # provably runs -> grants a token
                tokens[key] = tokens.get(key, 0) + 1
            # havoc producer: adversarially assumed absent -> contributes nothing
            continue
        # wait
        if status != "run":               # only report a definitely-running wait
            continue
        if tokens.get(key, 0) <= 0:
            if key in reported:
                continue
            reported.add(key)
            label = _eid_label(e.eid, sig_names)
            producers = sorted({pe.line for pe in body
                                if pe.kind == "set"
                                and (pe.p, pe.q, _canon_eid(pe.eid)) == key})
            rev_key = (e.q, e.p, _canon_eid(e.eid))
            primed_rev = any((x.p, x.q, _canon_eid(x.eid)) == rev_key
                             for x in pro if x.kind == "set")
            hint = (f"the prologue primes only the reverse direction "
                    f"({e.q}->{e.p}); " if primed_rev else
                    "this direction is never primed in the prologue; ")
            prod = (f"its only producer(s) in the loop are at line(s) {producers} "
                    f"(guarded and/or loop-carried, so absent on the first "
                    f"iteration)" if producers else
                    "it has no producer in the loop body")
            diags.append(Diag(
                "error", "PRIME_UNDERFLOW",
                f"scope {scope!r}: wait_flag({e.p!r}, {e.q!r}, {label}) at L{e.line} "
                f"runs unconditionally on the first executed iteration but no token "
                f"is available — {hint}{prod}. Cores whose task range starts in a "
                f"phase where the guarded producer is skipped will hang here "
                f"(AICore timeout). Fix: prime set_flag({e.p!r}, {e.q!r}, {label}) "
                f"in the prologue, make the wait conditional on the same guard as "
                f"its producer, or guarantee the loop starts in the producing phase.",
                line=e.line))
        else:
            tokens[key] = tokens[key] - 1
    return diags


def analyze_source(source: str) -> list[Diag]:
    tree = ast.parse(source)
    # Manual-sync only: with AUTO_SYNC the compiler inserts/primes the fences.
    if not find_auto_sync_disabled(tree):
        return []
    consts = collect_module_constants(tree)
    ev = ConstEvaluator(consts)
    space = build_space_map(tree)
    slot = build_slot_renames(tree, ev)
    sig_names: dict = {v: nm for nm, v in consts.items()
                       if isinstance(v, int) and nm.startswith("SIG_")}

    diags: list[Diag] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        scope = _scope_name(node)
        if scope is None:
            continue
        pro_stmts, loop = _split_loop(node.body)
        if loop is None:
            continue
        pro_ev = ScopeLinearizer(ev, space, slot).run(pro_stmts)
        body_ev = ScopeLinearizer(ev, space, slot).run(loop.body)
        diags += detect_prime_underflow(scope, pro_ev, body_ev, loop,
                                        consts, ev, sig_names)
    diags.sort(key=lambda d: (d.line, d.code))
    return diags


def analyze_file(path: str) -> list[Diag]:
    with open(path, "r", encoding="utf-8") as f:
        return analyze_source(f.read())


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tier-3 first-iteration credit-underflow (prime-balance) check")
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
            total += sum(1 for d in diags if d.severity == "error")
        elif not args.quiet:
            print(f"{path}: OK (no first-iteration credit underflow found)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
