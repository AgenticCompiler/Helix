from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from parser_base import find_newest_csv


# ============================================================================
# Hardware parameters — built-in profiles + custom override
# ============================================================================

@dataclass(frozen=True)
class RooflineHardware:
    """Peak hardware limits for roofline analysis."""
    label: str
    peak_tflops: float       # FP16 TFLOPS
    peak_bw_gbs: float       # HBM bandwidth (GB/s)

    @property
    def peak_gflops(self) -> float:
        return self.peak_tflops * 1000.0

    @property
    def ridge_ai(self) -> float:
        """Arithmetic intensity at the compute/memory boundary (FLOPs/Byte)."""
        return self.peak_gflops / self.peak_bw_gbs

    # -- built-in registry -------------------------------------------------

    _BUILTIN: dict[str, RooflineHardware] = {}  # populated below

    @classmethod
    def register(cls, name: str, label: str, peak_tflops: float, peak_bw_gbs: float) -> RooflineHardware:
        hw = cls(label=label, peak_tflops=peak_tflops, peak_bw_gbs=peak_bw_gbs)
        cls._BUILTIN[name] = hw
        return hw

    @classmethod
    def from_name(cls, name: str) -> RooflineHardware:
        """Look up a built-in hardware profile by name (case-insensitive)."""
        key = name.lower()
        if key not in cls._BUILTIN:
            available = ", ".join(sorted(cls._BUILTIN))
            raise ValueError(f"Unknown hardware '{name}'. Available: {available}")
        return cls._BUILTIN[key]

    @classmethod
    def get_default(cls) -> RooflineHardware:
        return cls._BUILTIN[_DEFAULT_HARDWARE]

    @classmethod
    def list_builtin(cls) -> dict[str, RooflineHardware]:
        return dict(cls._BUILTIN)


# -- Ascend 910B series (A2 platform) ----------------------------------------
RooflineHardware.register("910b1", "Ascend 910B1 FP16", peak_tflops=414.0, peak_bw_gbs=1600.0)
RooflineHardware.register("910b2", "Ascend 910B2 FP16", peak_tflops=376.0, peak_bw_gbs=1600.0)
RooflineHardware.register("910b3", "Ascend 910B3 FP16", peak_tflops=313.0, peak_bw_gbs=1600.0)
RooflineHardware.register("910b4", "Ascend 910B4 FP16", peak_tflops=280.0, peak_bw_gbs=800.0)

# -- Ascend 910C series (A3 platform) ----------------------------------------
RooflineHardware.register("910c",  "Ascend 910C  FP16", peak_tflops=800.0, peak_bw_gbs=3200.0)

# Default
_DEFAULT_HARDWARE = "910b3"


# ============================================================================
# CSV column helpers — flexible across msprof versions
# ============================================================================

def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_col(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in fieldnames:
            return col
    return None


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


# ============================================================================
# CSV file discovery — mirrors MsprofParser.parse() contract
# ============================================================================

def _require_csv(artifacts_dir: Path, prefix: str) -> Path:
    """Find CSV via find_newest_csv or raise a descriptive error."""
    csv_path = find_newest_csv(artifacts_dir, prefix)
    if csv_path is None:
        raise FileNotFoundError(
            f"No {prefix} CSV found in {artifacts_dir}"
        )
    return csv_path


# ============================================================================
# Core roofline analysis
# ============================================================================

def compute_roofline(
    artifacts_dir: Path,
    hardware: RooflineHardware | None = None,
) -> dict[str, Any]:
    """Run roofline analysis on a single msprof artifacts directory.

    Uses ``find_newest_csv`` (same contract as ``MsprofParser.parse``) to
    locate *OpBasicInfo*, *ArithmeticUtilization*, and *Memory* CSVs.

    Args:
        artifacts_dir: Path to the msprof output directory (e.g.
            ``PROF_*/mindstudio_profiler_output/``).
        hardware: Optional hardware parameters (defaults to 910B3 FP16).

    Returns:
        A dict with arithmetic intensity, real/theoretical GFLOPS, roof %,
        and raw intermediate values (prefixed with ``_``).
    """
    if hardware is None:
        hardware = RooflineHardware.get_default()

    # -- locate CSVs via find_newest_csv (same pattern as MsprofParser) ----
    basic_csv = _require_csv(artifacts_dir, "OpBasicInfo")
    arith_csv = _require_csv(artifacts_dir, "ArithmeticUtilization")
    mem_csv = _require_csv(artifacts_dir, "Memory")

    basic_rows = _read_csv_rows(basic_csv)
    arith_rows = _read_csv_rows(arith_csv)
    mem_rows = _read_csv_rows(mem_csv)

    if not basic_rows:
        raise ValueError(f"OpBasicInfo CSV is empty: {basic_csv}")
    if not arith_rows:
        raise ValueError(f"ArithmeticUtilization CSV is empty: {arith_csv}")
    if not mem_rows:
        raise ValueError(f"Memory CSV is empty: {mem_csv}")

    # -- OpBasicInfo: Block Dim, Task Duration ----------------------------
    basic = basic_rows[0]
    basic_fields = list(basic.keys())

    task_dur_col = _find_col(basic_fields, ("Task Duration(us)", "task_time(us)"))
    block_dim_col = _find_col(basic_fields, ("Block Dim", "Mix Block Dim"))

    task_dur_us = _safe_float(basic[task_dur_col]) if task_dur_col else 0.0
    block_dim = int(_safe_float(basic[block_dim_col]) or 0) if block_dim_col else 0
    kernel_runtime_s = (task_dur_us or 0.0) / 1e6

    # -- ArithmeticUtilization: FLOPs, vec ratio --------------------------
    arith_fields = list(arith_rows[0].keys())

    cube_col = _find_col(arith_fields, ("aic_cube_fops", "cube_fops"))
    vec_fops_col = _find_col(arith_fields, ("aiv_vec_fops", "vec_fops"))
    vec_ratio_col = _find_col(arith_fields, ("aiv_vec_ratio", "vec_ratio"))

    total_flops = 0.0
    for row in arith_rows:
        if cube_col:
            total_flops += _safe_float(row.get(cube_col, "")) or 0.0
        if vec_fops_col:
            total_flops += _safe_float(row.get(vec_fops_col, "")) or 0.0

    vec_ratio = 0.0
    if vec_ratio_col:
        ratios = [
            _safe_float(row.get(vec_ratio_col, "")) or 0.0
            for row in arith_rows
        ]
        vec_ratio = sum(ratios) / len(ratios) if ratios else 0.0

    # -- Memory: GM ↔ UB data volumes, bandwidth --------------------------
    mem_fields = list(mem_rows[0].keys())

    read_col = _find_col(mem_fields, ("GM_to_UB_datas(KB)", "gm_to_ub_datas(KB)", "GM Read(KB)"))
    write_col = _find_col(mem_fields, ("UB_to_GM_datas(KB)", "ub_to_gm_datas(KB)", "GM Write(KB)"))
    read_bw_col = _find_col(mem_fields, ("aiv_gm_to_ub_bw(GB/s)", "gm_to_ub_bw(GB/s)", "GM Read BW(GB/s)"))
    write_bw_col = _find_col(mem_fields, ("aiv_ub_to_gm_bw(GB/s)", "ub_to_gm_bw(GB/s)", "GM Write BW(GB/s)"))
    read_usage_col = _find_col(mem_fields, ("GM_to_UB_bw_usage_rate(%)", "gm_to_ub_bw_usage(%)"))
    write_usage_col = _find_col(mem_fields, ("UB_to_GM_bw_usage_rate(%)", "ub_to_gm_bw_usage(%)"))

    total_gm_kb = 0.0
    dma_read_bw_sum = 0.0
    dma_write_bw_sum = 0.0
    read_usage_sum = 0.0
    write_usage_sum = 0.0
    for row in mem_rows:
        if read_col:
            total_gm_kb += _safe_float(row.get(read_col, "")) or 0.0
        if write_col:
            total_gm_kb += _safe_float(row.get(write_col, "")) or 0.0
        if read_bw_col:
            dma_read_bw_sum += _safe_float(row.get(read_bw_col, "")) or 0.0
        if write_bw_col:
            dma_write_bw_sum += _safe_float(row.get(write_bw_col, "")) or 0.0
        if read_usage_col:
            read_usage_sum += _safe_float(row.get(read_usage_col, "")) or 0.0
        if write_usage_col:
            write_usage_sum += _safe_float(row.get(write_usage_col, "")) or 0.0

    total_gm_byte = total_gm_kb * 1024.0
    row_count = len(mem_rows) if mem_rows else 1
    dma_read_bw = dma_read_bw_sum / row_count
    dma_write_bw = dma_write_bw_sum / row_count
    read_usage_avg = read_usage_sum / row_count
    write_usage_avg = write_usage_sum / row_count

    # -- Roofline metrics -------------------------------------------------
    if total_gm_byte > 0:
        ai_kernel = total_flops / total_gm_byte
    else:
        ai_kernel = 0.0

    if kernel_runtime_s > 0:
        p_real_gflops = total_flops / kernel_runtime_s / 1e9
    else:
        p_real_gflops = 0.0

    p_bw_limit = hardware.peak_bw_gbs * ai_kernel
    p_theory = min(p_bw_limit, hardware.peak_gflops)

    roof_pct = (p_real_gflops / p_theory * 100.0) if p_theory > 0 else 0.0

    return {
        "hardware": hardware.label,
        "peak_tflops": hardware.peak_tflops,
        "peak_bw_gbs": hardware.peak_bw_gbs,
        "ridge_ai": hardware.ridge_ai,
        "task_duration_us": task_dur_us,
        "block_dim": block_dim,
        "ai": ai_kernel,
        "theoretical_gflops": p_theory,
        "real_gflops": p_real_gflops,
        "roof_pct": roof_pct,
        "vec_ratio_pct": vec_ratio * 100.0,
        "dma_read_bw_gbs": dma_read_bw,
        "dma_write_bw_gbs": dma_write_bw,
        "gm_to_ub_bw_usage_pct": read_usage_avg,
        "ub_to_gm_bw_usage_pct": write_usage_avg,
        "_total_flops": total_flops,
        "_total_gm_byte": total_gm_byte,
        "_p_bw_limit": p_bw_limit,
    }


# ============================================================================
# Batch analysis
# ============================================================================

def analyze_rounds(
    round_dirs: dict[str, Path],
    hardware: RooflineHardware | None = None,
) -> list[dict[str, Any]]:
    """Run roofline analysis on multiple rounds.

    Args:
        round_dirs: Mapping of {label: artifacts_dir}.
        hardware: Optional hardware parameters.

    Returns:
        List of result dicts (one per round), sorted by label.
    """
    results: list[dict[str, Any]] = []
    for label in sorted(round_dirs):
        result = compute_roofline(round_dirs[label], hardware=hardware)
        result["label"] = label
        results.append(result)
    return results


# ============================================================================
# Bottleneck diagnosis
# ============================================================================

def diagnose(result: dict[str, Any], hardware: RooflineHardware | None = None) -> str:
    """Return a human-readable bottleneck diagnosis string."""
    if hardware is None:
        hardware = RooflineHardware.get_default()

    lines = [
        f"--- {result.get('label', 'unknown')} Bottleneck Diagnosis ---",
        f"AI = {result['ai']:.2f} FLOPs/Byte  |  Ridge = {hardware.ridge_ai:.1f} FLOPs/Byte",
        f"Real GFLOPS = {result['real_gflops']:.1f}  |  Theoretical = {result['theoretical_gflops']:.1f}  |  Roof% = {result['roof_pct']:.1f}%",
        f"vec_ratio = {result['vec_ratio_pct']:.1f}%  |  DMA read BW = {result['dma_read_bw_gbs']:.2f} GB/s  |  DMA write BW = {result['dma_write_bw_gbs']:.2f} GB/s",
    ]

    ai = result["ai"]
    roof_pct = result["roof_pct"]
    vec_pct = result["vec_ratio_pct"]

    if ai < hardware.ridge_ai and roof_pct > 70:
        lines.append(
            "Bottleneck: MEMORY-BOUND (bandwidth limited). "
            "Optimize: data reuse / tiling / UB cache."
        )
    elif ai >= hardware.ridge_ai and vec_pct > 75:
        lines.append(
            "Bottleneck: COMPUTE-BOUND (compute limited). "
            "Optimize: increase block size / Cube vectorization."
        )
    else:
        lines.append(
            f"Bottleneck: Pipeline latency bound (Roof={roof_pct:.0f}%, "
            f"DMA_BW={result['dma_read_bw_gbs']:.1f} GB/s). "
            "Optimize: reduce scalar ratio / increase DMA transfer size / reduce block count."
        )

    return "\n".join(lines)


# ============================================================================
# Roofline plot
# ============================================================================

def plot_roofline(
    results: list[dict[str, Any]],
    output_path: str | Path,
    hardware: RooflineHardware | None = None,
    *,
    title: str | None = None,
) -> Path:
    """Generate a roofline plot and save to *output_path*."""
    if hardware is None:
        hardware = RooflineHardware.get_default()

    output_path = Path(output_path)

    plt.figure(figsize=(10, 6))

    # -- Roofline curve ---------------------------------------------------
    ai_range = np.logspace(-1, 4, 200)
    roof_perf = np.minimum(hardware.peak_bw_gbs * ai_range, hardware.peak_gflops)
    plt.loglog(ai_range, roof_perf, "k-", lw=2.5, label="Roofline Hardware Limit")
    plt.scatter(
        hardware.ridge_ai, hardware.peak_gflops,
        c="red", s=90, zorder=5,
        label=f"Ridge AI={hardware.ridge_ai:.1f}",
    )
    plt.axvline(x=hardware.ridge_ai, ls="--", c="gray", alpha=0.6)

    # -- Data points ------------------------------------------------------
    colors = plt.get_cmap('viridis')(np.linspace(0.2, 0.9, max(len(results), 1)))
    for idx, r in enumerate(results):
        label = r.get("label", f"round-{idx}")
        plt.scatter(
            r["ai"], r["real_gflops"],
            s=100, zorder=6, color=colors[idx],
            label=f"{label} (AI={r['ai']:.2f})",
        )

    # -- Annotations ------------------------------------------------------
    plt.text(
        hardware.ridge_ai / 8, hardware.peak_gflops / 10,
        "Memory-Bound\nRegion", fontsize=11, color="blue",
    )
    plt.text(
        hardware.ridge_ai * 6, hardware.peak_gflops / 2,
        "Compute-Bound\nRegion", fontsize=11, color="green",
    )

    plt.xlabel("Arithmetic Intensity FLOPs/Byte (log)")
    plt.ylabel("Perf GFLOPS/s (log)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=9)
    plt.title(title or f"{hardware.label} Roofline")
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


# ============================================================================
# Output formatters
# ============================================================================

def print_single(result: dict[str, Any]) -> None:
    """Print a detailed single-artifact roofline summary."""
    lines = [
        "=" * 60,
        f"Hardware:  {result['hardware']}",
        f"Peak:      {result['peak_tflops']} TFLOPS  |  {result['peak_bw_gbs']} GB/s  |  Ridge AI = {result['ridge_ai']:.1f}",
        "-" * 60,
        f"  Task Duration     : {result['task_duration_us']:.1f} us",
        f"  Block Dim         : {result['block_dim']}",
        f"  Arithmetic Intensity : {result['ai']:.2f} FLOPs/Byte",
        f"  Theoretical GFLOPS  : {result['theoretical_gflops']:.1f}",
        f"  Real GFLOPS         : {result['real_gflops']:.1f}",
        f"  Roof %              : {result['roof_pct']:.1f}%",
        f"  Vector Ratio        : {result['vec_ratio_pct']:.1f}%",
        f"  DMA Read BW         : {result['dma_read_bw_gbs']:.2f} GB/s",
        f"  DMA Write BW        : {result['dma_write_bw_gbs']:.2f} GB/s",
        f"  GM→UB BW Usage      : {result['gm_to_ub_bw_usage_pct']:.1f}%",
        f"  UB→GM BW Usage      : {result['ub_to_gm_bw_usage_pct']:.1f}%",
        "=" * 60,
    ]
    print("\n".join(lines))


def compare_roofline(
    baseline_dir: Path,
    target_dir: Path,
    *,
    hardware: RooflineHardware | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare roofline metrics between two msprof artifacts directories."""
    if hardware is None:
        hardware = RooflineHardware.get_default()

    baseline = compute_roofline(baseline_dir, hardware=hardware)
    baseline["label"] = "baseline"
    target = compute_roofline(target_dir, hardware=hardware)
    target["label"] = "target"

    def _delta_str(base_val: float, tgt_val: float) -> str:
        if base_val == 0:
            return "  N/A"
        delta = (tgt_val - base_val) / base_val * 100.0
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.1f}%"

    fields = [
        ("Task Duration(us)", "task_duration_us", "{:.1f}", True),
        ("Block Dim",         "block_dim",         "{}",    False),
        ("AI",                "ai",                "{:.2f}", False),
        ("Theoretical GFLOPS","theoretical_gflops","{:.1f}", False),
        ("Real GFLOPS",       "real_gflops",       "{:.1f}", False),
        ("Roof %",            "roof_pct",          "{:.1f}", False),
        ("vec_ratio %",       "vec_ratio_pct",     "{:.1f}", False),
        ("DMA Read BW",       "dma_read_bw_gbs",   "{:.2f}", False),
        ("DMA Write BW",      "dma_write_bw_gbs",  "{:.2f}", False),
        ("GM→UB BW Usage %",  "gm_to_ub_bw_usage_pct", "{:.1f}", False),
        ("UB→GM BW Usage %",  "ub_to_gm_bw_usage_pct", "{:.1f}", False),
    ]

    header = f"{'Metric':<22} {'Baseline':>12} {'Target':>12} {'Delta':>10}"
    sep = "-" * 58
    print("=" * 58)
    print(f"Roofline Comparison: {baseline_dir.name}  vs  {target_dir.name}")
    print(sep)
    print(header)
    print(sep)
    for label, key, fmt, lower_is_better in fields:
        b_val = baseline[key]
        t_val = target[key]
        if isinstance(b_val, float):
            b_str = fmt.format(b_val)
            t_str = fmt.format(t_val)
            delta = _delta_str(b_val, t_val)
        else:
            b_str = fmt.format(b_val)
            t_str = fmt.format(t_val)
            delta = "—"

        if lower_is_better and isinstance(b_val, (int, float)) and isinstance(t_val, (int, float)):
            if t_val < b_val:
                delta += " ✓"
            elif t_val > b_val:
                delta += " ✗"
        elif not lower_is_better and isinstance(b_val, (int, float)) and isinstance(t_val, (int, float)):
            if t_val > b_val:
                delta += " ✓"
            elif t_val < b_val:
                delta += " ✗"

        print(f"{label:<22} {b_str:>12} {t_str:>12} {delta:>10}")

    print("=" * 58)
    print(f"\n{diagnose(baseline, hardware=hardware)}")
    print(f"\n{diagnose(target, hardware=hardware)}")

    return baseline, target


def print_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted table of roofline results."""
    header = f"{'Round':<16} {'Dur(us)':>8} {'AI':>6} {'Real GFLOPS':>11} {'Roof%':>7} {'vec%':>6} {'DMA_R BW':>9}"
    sep = "-" * 65
    print("=" * 65)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.get('label', '?'):<16} "
            f"{r['task_duration_us']:>8.1f} "
            f"{r['ai']:>6.2f} "
            f"{r['real_gflops']:>11.1f} "
            f"{r['roof_pct']:>6.1f}% "
            f"{r['vec_ratio_pct']:>5.1f}% "
            f"{r['dma_read_bw_gbs']:>8.2f}"
        )


# ============================================================================
# Shared input-mode argument group — reused by all subcommands
# ============================================================================

def _add_input_args(subparser) -> None:
    """Add the mutually exclusive ``--artifacts-dir | --compare | --discover`` group."""
    group = subparser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--artifacts-dir", type=Path, dest="artifacts_dir",
        help="Single msprof artifacts directory (e.g. OPPROF_XXX / PROF_*/mindstudio_profiler_output/)",
    )
    group.add_argument(
        "--compare", nargs=2, type=Path, dest="compare_dirs",
        metavar=("BASELINE", "TARGET"),
        help="Two artifacts directories to compare",
    )
    group.add_argument(
        "--discover", type=Path, dest="discover_dir",
        help="Parent directory to scan for opt-round-* subdirectories",
    )


def _add_hardware_arg(subparser) -> None:
    """Add the ``--hardware`` argument shared by all subcommands."""
    available = ", ".join(sorted(RooflineHardware.list_builtin()))
    subparser.add_argument(
        "--hardware", default=_DEFAULT_HARDWARE,
        help=f"Hardware profile name (available: {available}; default: {_DEFAULT_HARDWARE})",
    )


# ============================================================================
# Input-mode dispatch helpers
# ============================================================================

def _resolve_hardware(args) -> RooflineHardware:
    return RooflineHardware.from_name(args.hardware)


def discover_rounds(
    parent_dir: str | Path,
    *,
    round_prefix: str = "opt-round-",
    glob_suffix: str = "msprof_op_case*/OPPROF_*",
) -> dict[str, Path]:
    """Scan *parent_dir* for round directories and locate their msprof artifacts."""
    import glob as _glob

    parent = Path(parent_dir)
    round_specs: dict[str, Path] = {}
    for name in sorted(os.listdir(parent)):
        full = parent / name
        if not full.is_dir() or not name.startswith(round_prefix):
            continue
        pattern = str(full / glob_suffix)
        matches = _glob.glob(pattern)
        if matches:
            round_specs[name] = Path(matches[0])
    return round_specs


# ============================================================================
# Subcommand handlers
# ============================================================================

def _handle_roofline(args, hardware: RooflineHardware) -> None:
    """Handle ``roofline`` subcommand."""
    if args.artifacts_dir:
        result = compute_roofline(args.artifacts_dir, hardware=hardware)
        result["label"] = args.artifacts_dir.name
        print_single(result)
        print(f"\n{diagnose(result, hardware=hardware)}")
        if getattr(args, "plot", False):
            out_png = args.artifacts_dir / "roofline.png"
            plot_roofline([result], out_png, hardware=hardware)
            print(f"\nSaved: {out_png}")

    elif args.compare_dirs:
        baseline_dir, target_dir = args.compare_dirs
        baseline_r, target_r = compare_roofline(baseline_dir, target_dir, hardware=hardware)
        if getattr(args, "plot", False):
            out_png = target_dir / "roofline_compare.png"
            plot_roofline(
                [baseline_r, target_r], out_png, hardware=hardware,
                title=f"Roofline: {baseline_dir.name} vs {target_dir.name}",
            )
            print(f"\nSaved: {out_png}")

    elif args.discover_dir:
        rounds = discover_rounds(args.discover_dir)
        if not rounds:
            print(f"No round directories found under: {args.discover_dir}", file=__import__("sys").stderr)
            raise SystemExit(1)
        results = analyze_rounds(rounds, hardware=hardware)
        print_table(results)
        if getattr(args, "plot", False):
            out_png = args.discover_dir / "roofline.png"
            plot_roofline(results, out_png, hardware=hardware)
            print(f"\nSaved: {out_png}")
        if results:
            print(f"\n{diagnose(results[-1], hardware=hardware)}")


def _print_stub_header(command: str, hardware: RooflineHardware) -> None:
    print(f"[{command}] hardware: {hardware.label} ({hardware.peak_tflops} TFLOPS, {hardware.peak_bw_gbs} GB/s)")
    print()


def _handle_pipeline(args, hardware: RooflineHardware) -> None:
    """Handle ``pipeline`` subcommand (stub)."""
    _print_stub_header("pipeline", hardware)
    if args.artifacts_dir:
        print("Pipeline analysis is not yet implemented.")
        print(f"Would analyze: {args.artifacts_dir}")
    elif args.compare_dirs:
        print("Pipeline comparison is not yet implemented.")
        print(f"Would compare: {args.compare_dirs[0]} vs {args.compare_dirs[1]}")
    elif args.discover_dir:
        print("Pipeline batch analysis is not yet implemented.")
        print(f"Would scan: {args.discover_dir}")


def _handle_summary(args, hardware: RooflineHardware) -> None:
    """Handle ``summary`` subcommand (stub)."""
    _print_stub_header("summary", hardware)
    if args.artifacts_dir:
        print("Summary analysis is not yet implemented.")
        print(f"Would analyze: {args.artifacts_dir}")
    elif args.compare_dirs:
        print("Summary comparison is not yet implemented.")
        print(f"Would compare: {args.compare_dirs[0]} vs {args.compare_dirs[1]}")
    elif args.discover_dir:
        print("Summary batch analysis is not yet implemented.")
        print(f"Would scan: {args.discover_dir}")


# ============================================================================
# CLI
# ============================================================================

def _build_cli():
    import argparse as _argparse

    p = _argparse.ArgumentParser(
        prog="msprof_op_parser",
        description="Ascend NPU msprof artifacts analysis.",
    )
    sub = p.add_subparsers(dest="command")

    # -- roofline ----------------------------------------------------------
    roofline = sub.add_parser("roofline", help="Arithmetic intensity & roofline analysis")
    _add_input_args(roofline)
    _add_hardware_arg(roofline)
    roofline.add_argument("--plot", action="store_true",
                          help="Generate a roofline plot (roofline only)")

    # -- pipeline (stub) ---------------------------------------------------
    pipeline = sub.add_parser("pipeline", help="Pipeline stage breakdown (AIC/AIV utilization)")
    _add_input_args(pipeline)
    _add_hardware_arg(pipeline)

    # -- summary (stub) ----------------------------------------------------
    summary = sub.add_parser("summary", help="Operator-level timing summary (hot ops, ratios)")
    _add_input_args(summary)
    _add_hardware_arg(summary)
    summary.add_argument("--top", type=int, default=10,
                         help="Number of top operators to show (default: 10)")

    # -- help --------------------------------------------------------------
    help_cmd = sub.add_parser("help", help="Show detailed usage with examples")

    return p, help_cmd


def _print_detailed_help(prog: str) -> None:
    """Print detailed usage help with examples."""
    builtin_hw = ", ".join(sorted(RooflineHardware.list_builtin()))
    print(f"""\
Usage: {prog} <command> [options]

Commands:
  roofline   Arithmetic intensity & roofline analysis
  pipeline   Pipeline stage breakdown (AIC/AIV utilization) [coming soon]
  summary    Operator-level timing summary (hot ops, ratios) [coming soon]

Input modes (shared by all commands, mutually exclusive):
  --artifacts-dir DIR       Single msprof artifacts directory
  --compare BASELINE TARGET Two artifacts directories to compare
  --discover DIR            Scan parent directory for opt-round-* subdirs

Shared options:
  --hardware NAME           Hardware profile (built-in: {builtin_hw}; default: {_DEFAULT_HARDWARE})

Roofline-only options:
  --plot                    Generate a roofline plot

Examples:
  {prog} roofline --artifacts-dir OPPROF_001
  {prog} roofline --artifacts-dir OPPROF_001 --plot --hardware 910c
  {prog} roofline --compare OPPROF_001 OPPROF_002
  {prog} roofline --compare OPPROF_001 OPPROF_002 --plot
  {prog} roofline --discover ./runs/
  {prog} pipeline --artifacts-dir OPPROF_001
  {prog} summary  --discover ./runs/ --top 5

Use "{prog} <command> -h" for command-specific options.\
""")


# ============================================================================
# __main__
# ============================================================================

if __name__ == "__main__":
    import sys

    parser, _help_cmd = _build_cli()  # noqa: F841 (help_cmd reserved for argparse)
    args = parser.parse_args()

    if args.command == "help":
        _print_detailed_help(parser.prog)
        raise SystemExit(0)

    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    hardware = _resolve_hardware(args)

    if args.command == "roofline":
        _handle_roofline(args, hardware)
    elif args.command == "pipeline":
        _handle_pipeline(args, hardware)
    elif args.command == "summary":
        _handle_summary(args, hardware)
    else:
        parser.print_help()
        raise SystemExit(1)
