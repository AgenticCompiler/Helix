"""Tier-2 deferred-credit ("late release") deadlock check for TileLang-Ascend.

Catches a loop-carried deadlock class that Tier 0 (flag-balance) and Tier 1
(intra-iteration happens-before) both miss, because the buggy and correct
kernels have *identical* flag structure -- same set/wait counts, same
capacities, same prologue priming. The only difference is the program-order
*position* of a buffer-reuse credit release.

Why a marked-graph cycle check is not enough
--------------------------------------------
The natural Tier-2 idea is a Petri-net / marked-graph zero-token-cycle search.
But for the motivating fixture (`mhc_post ..._defer_ws_post`) both the broken
and the correct version have the same channels with the same initial tokens; an
intra-scope zero-token-cycle search finds no cycle in *either*, and the
cross-core ring carries `RING` tokens of slack on every cube<->vector cycle, so
a pure flag-structure marked graph cannot separate them. Forward simulation of
the abstract credit model likewise does not deadlock. The deadlock is a
*timing* collapse: a primed buffer-reuse credit is released so late (after a
downstream store / cross-core handshake) that the effective pipeline depth drops
below what the cross-core ring needs, and the hardware ring starves.

What this checker actually detects (LATE_RELEASE)
-------------------------------------------------
The structural invariant that distinguishes the two files and matches the
confirmed root cause: a *primed* reuse credit `set_flag(p, q, id)` (the reverse
direction `(p, q, id)` is set in the prologue, i.e. it grants initial
permission) is the release that lets pipe `q` reload a buffer `B`. It should be
issued immediately after pipe `p`'s last read of `B`. If instead it is deferred
past one or more later cross-pipe handshakes on pipe `p` (a `set_flag` /
`wait_flag` of a *different* event, e.g. the output-store SIG_OUT), the credit
is held too long; on a cross-core-pipelined kernel that can collapse the ring
slack and deadlock.

This is a heuristic (held-too-long does not *always* deadlock), so findings are
warnings. It reuses the Tier-1 AST linearization (pipes + buffer read/write
sets) and needs no tilelang / TVM / NPU install.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from collections import Counter
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


_LOOP_FUNCS = ("serial", "Pipelined", "range", "grid")


def _split_loop(withbody) -> tuple[list, Optional[ast.For]]:
    """Split a scope body into (prologue stmts, main serial loop)."""
    pro: list = []
    for st in withbody:
        if (isinstance(st, ast.For) and isinstance(st.iter, ast.Call)
                and _attr_tail(st.iter.func) in _LOOP_FUNCS):
            return pro, st
        pro.append(st)
    return pro, None


def detect_late_release(scope: str, pro, body, cross_core: set) -> list[Diag]:
    """Flag primed buffer-reuse credits released past a downstream handshake,
    but only when the guarded buffer is reloaded from a *cross-core* buffer.

    Deferring a reuse credit past later handshakes is common and harmless in
    purely intra-core pipelines (e.g. a resident `q_l1` loaded once from GM). It
    only collapses cross-iteration slack -- and can deadlock -- when the buffer
    the credit gates is refilled from data the *other* subcore produces through
    the cross-core ring (here `acc_ub` <- `workspace_acc`, written by the cube).
    `cross_core` is the set of buffers written in one scope and read in another.
    """
    primed: Counter = Counter()
    for e in pro:
        if e.kind == "set":
            primed[(e.p, e.q, e.eid)] += 1

    diags: list[Diag] = []
    for (p, q, eid), cnt in primed.items():
        if cnt < 1:
            continue
        acquires = [e for e in body
                    if e.kind == "wait" and (e.p, e.q, e.eid) == (p, q, eid)]
        releases = [e for e in body
                    if e.kind == "set" and (e.p, e.q, e.eid) == (p, q, eid)]
        # Only a *dedicated* single-buffer reuse credit (exactly one acquire and
        # one release per iteration) is analyzable this way. A flag reused for
        # several buffers/handshakes in one iteration (multiple acquires or
        # releases, e.g. a shared `SIG_IO_UB` scratch fence) is multiplexed and
        # cannot be paired structurally -> skip rather than mispair and warn.
        if len(acquires) != 1 or len(releases) != 1:
            continue

        # Guarded buffer B: first op after the acquire on the *consumer* pipe q
        # that writes a buffer (the reload that the credit gates).
        acq = acquires[0]
        guarded: Optional[str] = None
        for e in body:
            if e.idx > acq.idx and e.pipe == q and e.kind == "op" and e.writes:
                # The reload must consume cross-core data for the deferral to
                # threaten the ring; otherwise it is a benign intra-core stall.
                if any(r in cross_core for r in e.reads):
                    guarded = e.writes[0]
                break
        if guarded is None:
            continue

        rel = releases[0]
        # Last read of B on the release pipe p before the release.
        last_read = None
        for e in body:
            if (e.idx < rel.idx and e.pipe == p and e.kind == "op"
                    and guarded in e.reads):
                if last_read is None or e.idx > last_read.idx:
                    last_read = e
        if last_read is None:
            continue

        # Intervening synchronization on pipe p between the last read and the
        # release that belongs to a *different* producer->consumer pipe pair.
        # Deferring a reuse credit past a sibling reuse credit on the SAME pipe
        # pair (e.g. releasing SIG_P right after SIG_KV, both MTE1->MTE2) is just
        # harmless release reordering. Deferring it past a handshake on another
        # pipe pair -- the output-store / next pipeline stage (e.g. V->MTE3
        # SIG_OUT) -- is what collapses the cross-iteration ring slack.
        interv = [e for e in body
                  if last_read.idx < e.idx < rel.idx and e.pipe == p
                  and e.kind in ("set", "wait")
                  and (e.p, e.q) != (p, q)]
        if not interv:
            continue

        names = ", ".join(sorted({f"{e.kind}({e.p}->{e.q},{e.eid})"
                                  for e in interv}))
        diags.append(Diag(
            "warning", "LATE_RELEASE",
            f"scope {scope!r}: primed reuse credit set_flag({p!r}, {q!r}, {eid}) "
            f"for buffer {guarded!r} is released at L{rel.line}, deferred past "
            f"{len(interv)} later handshake(s) [{names}] beyond its last read at "
            f"L{last_read.line}. Holding a primed buffer-reuse credit past a "
            f"downstream store/cross-core handshake collapses cross-iteration "
            f"pipeline slack and can deadlock the ring. Fix: move "
            f"set_flag({p!r}, {q!r}, {eid}) to immediately after L{last_read.line}.",
            line=rel.line))
    return diags


def analyze_source(source: str) -> list[Diag]:
    tree = ast.parse(source)
    # Manual-sync only: with TL_ASCEND_AUTO_SYNC the compiler places releases.
    if not find_auto_sync_disabled(tree):
        return []
    consts = collect_module_constants(tree)
    ev = ConstEvaluator(consts)
    space = build_space_map(tree)
    slot = build_slot_renames(tree, ev)

    # Per-scope full linearization, used both to identify cross-core buffers
    # (written in one scope, read in another) and to run the per-scope check.
    scopes: list[tuple[str, list, ast.For]] = []
    writes_in: dict[str, set] = {}
    reads_in: dict[str, set] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        scope = _scope_name(node)
        if scope is None:
            continue
        pro_stmts, loop = _split_loop(node.body)
        if loop is None:
            continue
        all_ev = ScopeLinearizer(ev, space, slot).run(node.body)
        for e in all_ev:
            if e.kind != "op":
                continue
            for b in e.writes:
                writes_in.setdefault(b, set()).add(scope)
            for b in e.reads:
                reads_in.setdefault(b, set()).add(scope)
        scopes.append((scope, pro_stmts, loop))

    cross_core = {b for b in writes_in
                  if any(s not in writes_in[b] for s in reads_in.get(b, set()))}

    diags: list[Diag] = []
    for scope, pro_stmts, loop in scopes:
        pro_ev = ScopeLinearizer(ev, space, slot).run(pro_stmts)
        body_ev = ScopeLinearizer(ev, space, slot).run(loop.body)
        diags += detect_late_release(scope, pro_ev, body_ev, cross_core)
    diags.sort(key=lambda d: (d.line, d.code))
    return diags


def analyze_file(path: str) -> list[Diag]:
    with open(path, "r", encoding="utf-8") as f:
        return analyze_source(f.read())


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tier-2 deferred-credit (late-release) deadlock check")
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
            print(f"{path}: OK (no late-release hazards found)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
