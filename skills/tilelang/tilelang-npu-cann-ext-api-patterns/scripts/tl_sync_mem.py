"""Tier-4: cross-vid shared-memory (wrong workspace) linter.

Detects SHARED_MAILBOX: an alloc_shared (L1/UB on-chip) buffer that receives
a V-pipe write from UB AND is V-pipe read back to UB — the signature of an
on-chip cross-lane mailbox.  The correct exchange path for dual-vid kernels is
a GM tensor argument (MTE3 write, MTE2 read), as specified by MCScan and the
AscendNPU-IR memory-hierarchy docs.

Constraint from the docs
------------------------
"Data can only be exchanged [between AIV cores] using global memory and/or
L2 cache."  (MCScan §3; AscendNPU-IR memory-hierarchy.md)

alloc_shared maps to L1 Buffer (cube path) or UB (vector path) — both are
per-AIV on-chip memories (TileLang-Ascend Programming Guide §4.1.1).  Writing
computed results from UB into an alloc_shared buffer and then reading that
buffer back into UB bypasses the GM/L2 exchange path and causes device error
507015 ("VEC supports illegal configurations in commands") when the reader is
a different AIV lane.

What the check looks for
------------------------
For each alloc_shared buffer B in a T.Kernel body:

  1. V-pipe write:  T.copy(src, B)  where src is NOT in GM
                    (i.e. src ∈ UB / L1, not an MTE2 staging load from GM)

  2. V-pipe read:   T.copy(B, dst)  where dst is NOT in L0A/L0B or GM
                    (i.e. dst ∈ UB, not an MTE1 cube feed or MTE3 store to GM)

If BOTH (1) and (2) exist for the same buffer → SHARED_MAILBOX error.
Exchange buffers must be GM tensor arguments, not alloc_shared.

This check does NOT require tracking vid==N/else guards, because the memory-
space violation is present regardless of the control-flow structure: any
alloc_shared buffer used as a write-from-UB / read-to-UB exchange should be
a GM tensor.

Correct vs. incorrect exchange patterns
----------------------------------------
  Correct   (MTE3 write + MTE2 read — GM exchange):
    T.copy(sxx, partial_ws[cid, lane, stat, 0:1])  # UB -> GM : MTE3
    T.barrier_all()
    T.copy(partial_ws[cid, other, stat, 0:1], tile_sum)  # GM -> UB : MTE2

  Incorrect (V-pipe write + V-pipe read — on-chip mailbox):
    T.copy(sxx, p1_sxx)       # UB -> L1 : V-pipe  <- FLAGGED (write)
    T.barrier_all()
    T.copy(p1_sxx, tile_sum)  # L1 -> UB : V-pipe  <- FLAGGED (read)

Ground truth
------------
  engram_gate_fwd_npu_debug.py   — SHARED_MAILBOX on p0_sxx, p0_skk, p0_sdot,
                                    p1_sxx, p1_skk, p1_sdot (all six partials
                                    use alloc_shared; gmws.py uses GM instead)
  engram_gate_fwd_npu_debug_gmws.py — clean (partial_ws is a GM parameter)

Note: p0_* buffers do not crash the kernel (vid==0 reads its own copy) but
ARE design violations — the correct pattern puts ALL cross-vid exchange in GM,
as gmws.py demonstrates.  The check correctly flags them to point at the fix.

What this check does NOT cover
-------------------------------
- alloc_shared buffers written from GM (MTE2 staging loads) — these are
  legitimate input staging buffers and are never flagged.
- alloc_shared buffers read to L0A/L0B (MTE1 cube feed) or to GM (MTE3 store)
  — these are normal cube / output paths and are never flagged.
- T.tile.* / reduce_* ops on alloc_shared (copy is the dominant exchange path;
  extend if the pattern widens).
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl_sync_lint import Diag, _attr_tail  # noqa: E402
from tl_sync_hb import ALLOC_SPACE, build_space_map  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _base_name(node: ast.expr) -> Optional[str]:
    """Unwrap subscripts/attributes to find the root buffer name."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


# --------------------------------------------------------------------------- #
# Copy operation record
# --------------------------------------------------------------------------- #

@dataclass
class CopyOp:
    buf: str          # the alloc_shared buffer involved
    mode: str         # "w" (B is dst) or "r" (B is src)
    peer_space: str   # memory space of the OTHER operand
    line: int


def _collect_copy_ops(
    body: list,
    shared_bufs: set[str],
    space: dict[str, str],
) -> list[CopyOp]:
    """Walk all statements and collect T.copy operations involving alloc_shared
    buffers.  The peer_space is "GM" when the other operand is a GM tensor
    parameter, "UB" / "L1" / etc. otherwise."""
    ops: list[CopyOp] = []
    for node in ast.walk(ast.Module(body=list(body), type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        if _attr_tail(node.func) != "copy" or len(node.args) < 2:
            continue
        src_name = _base_name(node.args[0])
        dst_name = _base_name(node.args[1])
        src_space = space.get(src_name, "GM") if src_name else "GM"
        dst_space = space.get(dst_name, "GM") if dst_name else "GM"

        if dst_name in shared_bufs:
            ops.append(CopyOp(dst_name, "w", src_space, node.lineno))
        if src_name in shared_bufs:
            ops.append(CopyOp(src_name, "r", dst_space, node.lineno))
    return ops


# --------------------------------------------------------------------------- #
# The check
# --------------------------------------------------------------------------- #

# Memory spaces that indicate a V-pipe write source (computed, not from GM).
# A write from GM to alloc_shared is a legitimate MTE2 staging load.
_NON_GM_SRC = frozenset({"UB", "L1", "L0A", "L0B", "L0C"})

# Memory spaces that indicate a V-pipe read destination (back into compute).
# A read from alloc_shared to L0A/L0B is a legitimate MTE1 cube feed.
# A read from alloc_shared to GM is a legitimate MTE3 store.
_NON_GM_NON_L0_DST = frozenset({"UB", "L1"})


def analyze_source(source: str) -> list[Diag]:
    tree = ast.parse(source)
    space = build_space_map(tree)

    # Identify alloc_shared buffers across the whole file.
    shared_bufs: set[str] = {
        node.targets[0].id
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and _attr_tail(node.value.func) in ("alloc_shared", "alloc_L1")
        )
    }

    if not shared_bufs:
        return []

    # Collect copy operations for all T.Kernel bodies.
    all_ops: list[CopyOp] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            is_kernel = any(
                isinstance(item.context_expr, ast.Call)
                and _attr_tail(item.context_expr.func) == "Kernel"
                for item in node.items
            )
            if is_kernel:
                all_ops += _collect_copy_ops(node.body, shared_bufs, space)

    # Group by buffer.
    by_buf: dict[str, list[CopyOp]] = {}
    for op in all_ops:
        by_buf.setdefault(op.buf, []).append(op)

    diags: list[Diag] = []
    for buf, ops in sorted(by_buf.items()):
        # V-pipe writes: written FROM UB/L1 (computed result stored to alloc_shared).
        # NOT a V-pipe write: written from GM (legitimate MTE2 staging load).
        vpipe_writes = [op for op in ops if op.mode == "w" and op.peer_space in _NON_GM_SRC]

        # V-pipe reads: read TO UB/L1 (alloc_shared used as compute source).
        # NOT a V-pipe read: read to L0A/L0B (MTE1 cube feed) or to GM (MTE3 store).
        vpipe_reads = [op for op in ops if op.mode == "r" and op.peer_space in _NON_GM_NON_L0_DST]

        if not vpipe_writes or not vpipe_reads:
            continue  # Not used as a mailbox — staging or output buffer; skip.

        wlines = sorted({op.line for op in vpipe_writes})
        rlines = sorted({op.line for op in vpipe_reads})
        diags.append(Diag(
            "error",
            "SHARED_MAILBOX",
            (
                f"alloc_shared buffer {buf!r} is written from UB via V-pipe "
                f"(line(s) {wlines}) and read back to UB via V-pipe "
                f"(line(s) {rlines}). "
                f"alloc_shared is per-AIV on-chip memory (L1/UB) — "
                f"using it as a cross-lane exchange buffer bypasses the GM/L2 "
                f"data path and causes device error 507015 "
                f"('VEC supports illegal configurations in commands'). "
                f"Fix: replace this buffer with a GM tensor argument "
                f"(e.g. partial_ws[cid, lane, stat, 0:1]); write via MTE3 "
                f"(T.copy(ub_src, gm_buf)) and read via MTE2 "
                f"(T.copy(gm_buf, ub_dst)) after T.barrier_all() or "
                f"T.wait_cross_flag(SEM_*)."
            ),
            line=min(wlines + rlines),
        ))

    diags.sort(key=lambda d: (d.line, d.code))
    return diags


def analyze_file(path: str) -> list[Diag]:
    with open(path, "r", encoding="utf-8") as f:
        return analyze_source(f.read())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tier-4 TileLang shared-mailbox (wrong workspace) linter"
    )
    parser.add_argument("files", nargs="+", help="kernel .py file(s) to lint")
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
            print(f"{path}: OK (no shared-mailbox violations)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
