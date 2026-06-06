import os
import utils
import extract_api_instr
from collections import defaultdict

FILE_NAMES = {
    "trace": "dataType_2_TRACE.json",
    "flow": "flows.json",
    "instr": "dataType_4_API_INSTR.json"
}

SWIM_LANE_INFO = {
    "ALL": "",
    "SCALAR": "startup",
    "SCALARLDST": "",
    "FLOWCTRL": "",
    "MTE1": "",
    "CUBE": "",
    "FIXP": "",
    "MTE2": "thread_state_iowait",
    "VECTOR": "cq_build_failed",
    "MTE3": "rail_response",
    "CACHEMISS": "thread_state_runnable",
    "USEMASK": "",
}

# name -> cname
SPECIAL_EVENTS = {
    "WAIT_FLAG": "yellow",
    "SET_FLAG": "black",
}

# scalar, mte2, vector, cube, mte3
TRACED_OVERLAP = [
    [["SCALAR"], ["MTE2"]],
    [["SCALAR"], ["VECTOR"]],
    [["SCALAR"], ["CUBE"]],
    [["SCALAR"], ["MTE3"]],
    [["MTE2"], ["VECTOR"]],
    [["MTE2"], ["CUBE"]],
    [["MTE2"], ["MTE3"]],
    [["VECTOR"], ["CUBE"]],
    [["VECTOR"], ["MTE3"]],
    [["CUBE"], ["MTE3"]],
    [["VECTOR", "CUBE"], ["SCALAR"]],
    [["VECTOR", "CUBE"], ["MTE2"]],
    [["VECTOR", "CUBE"], ["MTE3"]],
]


"""
{
    "swim_lane_name": {
        "core_name": [
            [start_time, end_time],
            ...
        ]
    }
}
"""
NON_IDLE_TIME_INTERVAL = {}
"""
{
    "core_name": xxx
}
"""
SIMULATOR_RUNTIME = {}
"""
{
    "swim_lane_name": {
        "core_name": {
            "event_name": [
                [start_time, end_time],
                ...
            ]
        }
    }
}
"""
SPECIAL_EVENT_TIME_INTERVAL = {}


def _sum_valid(v) -> int:
    if isinstance(v, list):
        return sum(x for x in v if x >= 0)
    return int(v) if v is not None and v >= 0 else 0


def _avg_valid(v) -> float:
    if isinstance(v, list):
        vals = [x for x in v if x >= 0]
        return sum(vals) / len(vals) if vals else 0.0
    return float(v) if v is not None and v >= 0 else 0.0


def _is_mte2_data_mover(src: str) -> bool:
    s_upper = src.upper()
    if "SET_FLAG" in s_upper or "WAIT_FLAG" in s_upper or "END_LABEL" in s_upper:
        return False
    if s_upper.startswith("MOV_SPR_XN") and "SPR:" in src and "Src:" not in src:
        return False
    if "LD" in s_upper or "LOAD" in s_upper:
        return True
    if "MOV_SRC_TO_DST_ALIGN" in s_upper:
        return True
    if "LDP_XI_XJ_XN" in s_upper:
        return True
    if s_upper.startswith("STI_XN"):
        return True
    return False


def analyze_pipe(instructions):
    pipe_instr = defaultdict(int)
    pipe_cycles = defaultdict(int)
    for instr in instructions:
        pipe = instr.get("Pipe", "UNKNOWN")
        pipe_instr[pipe] += 1
        pipe_cycles[pipe] += _sum_valid(instr.get("Cycles", 0))
    total_instr = sum(pipe_instr.values())
    total_cycles = sum(pipe_cycles.values())
    result = {"total_instr": total_instr, "total_cycles": total_cycles, "pipes": {}}
    for pipe in sorted(pipe_instr.keys()):
        result["pipes"][pipe] = {
            "instr": pipe_instr[pipe],
            "instr_pct": round(pipe_instr[pipe] / total_instr * 100, 1) if total_instr else 0,
            "cycles": pipe_cycles[pipe],
            "cycles_pct": round(pipe_cycles[pipe] / total_cycles * 100, 1) if total_cycles else 0,
        }
    return result


def analyze_ratios(pipe_result):
    p = pipe_result["pipes"]
    total_instr = pipe_result.get("total_instr", 1)
    total_cycles = pipe_result.get("total_cycles", 1)
    s_instr = p.get("SCALAR", {}).get("instr", 0)
    v_instr = p.get("VECTOR", {}).get("instr", 0)
    s_cycles = p.get("SCALAR", {}).get("cycles", 0)
    v_cycles = p.get("VECTOR", {}).get("cycles", 0)
    mte2_cycles = p.get("MTE2", {}).get("cycles", 0)
    mte2_instr = p.get("MTE2", {}).get("instr", 0)
    return {
        "SCALAR:VECTOR_instr": f"{s_instr}:{v_instr} = {round(s_instr / max(v_instr, 1), 1)}:1",
        "SCALAR:VECTOR_cycles": f"{s_cycles}:{v_cycles} = {round(s_cycles / max(v_cycles, 1), 1)}:1",
        "SCALAR_instr_pct": round(s_instr / total_instr * 100, 1) if total_instr else 0,
        "SCALAR_cycles_pct": round(s_cycles / total_cycles * 100, 1) if total_cycles else 0,
        "VECTOR_instr_pct": round(v_instr / total_instr * 100, 1) if total_instr else 0,
        "VECTOR_cycles_pct": round(v_cycles / total_cycles * 100, 1) if total_cycles else 0,
        "MTE2_instr_pct": round(mte2_instr / total_instr * 100, 1) if total_instr else 0,
        "MTE2_cycles_pct": round(mte2_cycles / total_cycles * 100, 1) if total_cycles else 0,
    }


def analyze_vector(instructions):
    vec_instrs = [i for i in instructions if i.get("Pipe") == "VECTOR"]
    ub_read, ub_write = 0, 0
    utils_list = []
    sources = defaultdict(int)
    for instr in vec_instrs:
        urc = _sum_valid(instr.get("UB Read Conflict", 0))
        uwc = _sum_valid(instr.get("UB Write Conflict", 0))
        ub_read += urc
        ub_write += uwc
        raw_util = instr.get("Vector Utilization Percentage", 0)
        if isinstance(raw_util, list):
            utils_list.extend([x for x in raw_util if x >= 0])
        elif raw_util >= 0:
            utils_list.append(float(raw_util))
        sources[str(instr.get("Source", ""))[:80]] += 1
    conflicts = []
    for instr in vec_instrs:
        urc = _sum_valid(instr.get("UB Read Conflict", 0))
        uwc = _sum_valid(instr.get("UB Write Conflict", 0))
        if urc > 0 or uwc > 0:
            conflicts.append((str(instr.get("Source", ""))[:80], urc, uwc,
                              round(_avg_valid(instr.get("Vector Utilization Percentage", 0)), 1)))
    conflicts.sort(key=lambda x: x[1] + x[2], reverse=True)
    return {
        "instr_count": len(vec_instrs),
        "ub_read_conflict": ub_read,
        "ub_write_conflict": ub_write,
        "ub_conflict_total": ub_read + ub_write,
        "util_pct_avg": round(sum(utils_list) / len(utils_list), 1) if utils_list else 0.0,
        "util_pct_min": round(min(utils_list), 1) if utils_list else 0.0,
        "util_pct_max": round(max(utils_list), 1) if utils_list else 0.0,
        "util_sample_count": len(utils_list),
        "top_sources": dict(sorted(sources.items(), key=lambda x: -x[1])[:15]),
        "top_conflicts": conflicts[:5],
    }


def analyze_mte2(instructions):
    mte2_instrs = [i for i in instructions if i.get("Pipe") == "MTE2"]
    all_pb = []
    data_movers = 0
    data_pb_values = []
    control_ops = 0
    mte2_sources = defaultdict(int)
    for instr in mte2_instrs:
        src = str(instr.get("Source", ""))
        pb = _sum_valid(instr.get("ProcessBytes", 0))
        if isinstance(instr.get("ProcessBytes"), list):
            all_pb.extend([x for x in instr.get("ProcessBytes", 0) if x >= 0])
        elif pb >= 0:
            all_pb.append(pb)
        mte2_sources[src[:80]] += 1
        if _is_mte2_data_mover(src):
            data_movers += 1
            if pb > 0:
                data_pb_values.append(pb)
        else:
            control_ops += 1
    return {
        "instr_count": len(mte2_instrs),
        "data_mover_count": data_movers,
        "control_op_count": control_ops,
        "pb_data_mover_avg": round(sum(data_pb_values) / len(data_pb_values), 0) if data_pb_values else 0,
        "pb_data_mover_min": min(data_pb_values) if data_pb_values else 0,
        "pb_data_mover_max": max(data_pb_values) if data_pb_values else 0,
        "pb_all_avg": round(sum(all_pb) / len(all_pb), 0) if all_pb else 0,
        "pb_all_max": max(all_pb) if all_pb else 0,
        "top_sources": dict(sorted(mte2_sources.items(), key=lambda x: -x[1])[:10]),
    }


def analyze_cube(instructions):
    cube_instrs = [i for i in instructions if i.get("Pipe") == "CUBE"]
    if not cube_instrs:
        return {"present": False}
    total_cycles = 0
    mmad_count = 0
    for instr in cube_instrs:
        src = str(instr.get("Source", ""))
        cyc = _sum_valid(instr.get("Cycles", 0))
        total_cycles += cyc
        if "MMAD" in src:
            mmad_count += 1
    return {"present": True, "instr_count": len(cube_instrs),
            "mmad_count": mmad_count, "total_cycles": total_cycles}


def analyze_flowctrl(instructions):
    wait_by_pipe = defaultdict(int)
    bar_by_pipe = defaultdict(int)
    for instr in instructions:
        pipe = instr.get("Pipe", "UNKNOWN")
        src = str(instr.get("Source", ""))
        cnt = _sum_valid(instr.get("Instructions Executed", 0))
        if not cnt:
            cnt = 1
        if "WAIT_FLAG" in src or "WAIT" in src.upper():
            wait_by_pipe[pipe] += cnt
        if "BAR" in src or "BARRIER" in src:
            bar_by_pipe[pipe] += cnt
    return {
        "wait_total": sum(wait_by_pipe.values()),
        "bar_total": sum(bar_by_pipe.values()),
        "wait_by_pipe": dict(wait_by_pipe),
        "bar_by_pipe": dict(bar_by_pipe),
    }


def analyze_trace(events):
    complete = [e for e in events if e.get("ph") == "X"]
    names = defaultdict(int)
    for e in complete:
        names[e.get("name", "UNKNOWN")] += 1
    top = sorted(names.items(), key=lambda x: -x[1])
    arith_names = ["SIGNEXT", "ADD", "MUL", "DIV", "SUB", "MADD"]
    arithmetic = {k: names.get(k, 0) for k in arith_names}
    arith_total = sum(arithmetic.values())
    arith_pct = round(arith_total / len(complete) * 100, 1) if complete else 0
    return {
        "total_events": len(events),
        "complete_events": len(complete),
        "top_events": top[:20],
        "arithmetic": arithmetic,
        "arithmetic_total": arith_total,
        "arithmetic_pct_of_complete": arith_pct,
    }


def analyze_scalar(instructions):
    scalar = [i for i in instructions if i.get("Pipe") == "SCALAR"]
    sources = defaultdict(int)
    for instr in scalar:
        sources[str(instr.get("Source", ""))[:80]] += 1
    return {"instr_count": len(scalar),
            "top_sources": dict(sorted(sources.items(), key=lambda x: -x[1])[:15])}


def compute_pipe_per_core(instructions, cores):
    """Per-core pipe instruction/cycle distribution from API_INSTR."""
    if not cores or not instructions:
        return {}
    result = {}
    for core_idx, core_name in enumerate(cores):
        pipe_instr = defaultdict(int)
        pipe_cycles = defaultdict(int)
        for instr in instructions:
            pipe = instr.get("Pipe", "UNKNOWN")
            pipe_instr[pipe] += 1
            cycles_raw = instr.get("Cycles", [])
            if isinstance(cycles_raw, list) and core_idx < len(cycles_raw):
                c = cycles_raw[core_idx]
                if c >= 0:
                    pipe_cycles[pipe] += c
        total_instr = sum(pipe_instr.values())
        total_cycles = sum(pipe_cycles.values())
        core_result = {}
        for pipe in sorted(pipe_instr.keys()):
            core_result[pipe] = {
                "instr": pipe_instr[pipe],
                "instr_pct": round(pipe_instr[pipe] / total_instr * 100, 1) if total_instr else 0,
                "cycles": pipe_cycles[pipe],
                "cycles_pct": round(pipe_cycles[pipe] / total_cycles * 100, 1) if total_cycles else 0,
            }
        result[core_name] = core_result
    return result


"""
泳道非空占比 = Time(泳道非空)/Time(total)
去除wait等信号占用时间
"""
def collect_non_idle_time(data):
    if "ts" not in data or "dur" not in data or "cname" not in data:
        return
    if data["tid"] not in SWIM_LANE_INFO and data["cname"] != SWIM_LANE_INFO[data["tid"]]:
        return
    if data["dur"] == 0:
        return
    if data["tid"] not in NON_IDLE_TIME_INTERVAL:
        NON_IDLE_TIME_INTERVAL[data["tid"]] = {}
    if data["pid"] not in NON_IDLE_TIME_INTERVAL[data["tid"]]:
        NON_IDLE_TIME_INTERVAL[data["tid"]][data["pid"]] = []
    NON_IDLE_TIME_INTERVAL[data["tid"]][data["pid"]].append(
        [data["ts"], data["ts"] + data["dur"]]
    )


def compute_non_idle_ratio():
    """
    summary格式：
    {
        "swim_lane_name": {
            "core_name": xxx
        }
    }
    """
    summary = {}
    for swim_lane_name in NON_IDLE_TIME_INTERVAL:
        summary[swim_lane_name] = {}
        for core_name in NON_IDLE_TIME_INTERVAL[swim_lane_name]:
            non_idle_time = 0
            for item in NON_IDLE_TIME_INTERVAL[swim_lane_name][core_name]:
                non_idle_time += item[1] - item[0]
            summary[swim_lane_name][core_name] = non_idle_time / SIMULATOR_RUNTIME[core_name]
    return summary


def dump_compute_non_idle_ratio(summary, per_core_pipe=None):
    """Modified: adds instr/cycles from API_INSTR to per-core pipe distribution."""
    text = "[Pipe Distribution Over Each Core]\n"
    transform = {}
    for key in summary:
        for key2 in summary[key]:
            if key2 not in transform:
                transform[key2] = {}
            transform[key2][key] = summary[key][key2]
    for key in sorted(transform.keys()):
        text += f"  [Pipe Distribution Over {key}]\n"
        for key2 in sorted(transform[key].keys()):
            dur_pct = transform[key][key2]
            if per_core_pipe and key in per_core_pipe:
                pd = per_core_pipe[key].get(key2, {})
                instr_val = pd.get("instr", "")
                instr_pct = pd.get("instr_pct", 0)
                cycles_val = pd.get("cycles", "")
                cycles_pct = pd.get("cycles_pct", 0)
                if instr_val != "":
                    text += f"    %({key2}): instr={instr_val} ({instr_pct:.1f}%)  " \
                            f"cycles={cycles_val} ({cycles_pct:.1f}%)  " \
                            f"dur={dur_pct:.2%}\n"
                else:
                    text += f"    %({key2}): dur={dur_pct:.2%}\n"
            else:
                text += f"    %({key2}): {dur_pct:.2%}\n"
        text += "\n"
    return text


def compute_avg_non_idle_ratio(summary):
    avg_summary = {}
    for swim_lane_name in summary:
        avg_summary[swim_lane_name] = sum(summary[swim_lane_name].values()) / len(summary[swim_lane_name])
    return avg_summary


def collapse_time_interval(time_intervals):
    time_intervals.sort(key=lambda x: x[0])
    merged = [time_intervals[0]]
    for current in time_intervals[1:]:
        prev_end = merged[-1][1]
        current_start, current_end = current
        if current_start <= prev_end:
            merged[-1][1] = max(prev_end, current_end)
        else:
            merged.append(current)
    return merged


def collapse_time_interval_over_multi_swimlanes(swim_lane_names):
    if len(swim_lane_names) == 1:
        if swim_lane_names[0] not in NON_IDLE_TIME_INTERVAL:
            return {}
        return NON_IDLE_TIME_INTERVAL[swim_lane_names[0]]
    collection = {}
    for swim_lane_name in swim_lane_names:
        if swim_lane_name not in NON_IDLE_TIME_INTERVAL:
            continue
        for core_name in NON_IDLE_TIME_INTERVAL[swim_lane_name]:
            if core_name not in collection:
                collection[core_name] = []
            collection[core_name] += NON_IDLE_TIME_INTERVAL[swim_lane_name][core_name]
    for core_name in collection:
        collection[core_name] = collapse_time_interval(collection[core_name])
    return collection


def compute_sum(time_intervals):
    sum = 0
    for item in time_intervals:
        sum += item[1] - item[0]
    return sum


def or_key(swim_lane_name):
    text = "+".join(swim_lane_name)
    if len(swim_lane_name) > 1:
        return f"({text})"
    return text


def compute_key(swim_lane_name1, swim_lane_name2):
    text1 = or_key(swim_lane_name1)
    text2 = or_key(swim_lane_name2)
    return (f"{text1}&{text2}/{text1}", f"{text1}&{text2}/{text2}")


def dump_overlap_ratio(summary):
    text = "[Pipe Overlap Ratio Over Each Core]\n"
    transform = {}
    for key in summary:
        for key2 in summary[key]:
            if key2 not in transform:
                transform[key2] = {}
            transform[key2][key] = summary[key][key2]
    for key in sorted(transform.keys()):
        text += f"  [Pipe Overlap Ratio Of {key}]\n"
        for key2 in sorted(transform[key].keys()):
            text += f"    %({key2}): {transform[key][key2]:.2%}\n"
        text += "\n"
    return text


def compute_overlap_ratio():
    overlap_ratios = {}
    for pair in TRACED_OVERLAP:
        swim_lane_name1 = pair[0]
        swim_lane_name2 = pair[1]
        time_intervals1 = collapse_time_interval_over_multi_swimlanes(swim_lane_name1)
        time_intervals2 = collapse_time_interval_over_multi_swimlanes(swim_lane_name2)
        key1, key2 = compute_key(swim_lane_name1, swim_lane_name2)
        for core_name in time_intervals1:
            if core_name not in time_intervals2:
                continue
            overlaps = []
            i, j = 0, 0
            while i < len(time_intervals1[core_name]) and j < len(time_intervals2[core_name]):
                start1, end1 = time_intervals1[core_name][i]
                start2, end2 = time_intervals2[core_name][j]
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                if overlap_start <= overlap_end:
                    overlaps.append([overlap_start, overlap_end])
                if end1 < end2:
                    i += 1
                else:
                    j += 1
            overlap_time = compute_sum(overlaps)
            base_time1 = compute_sum(time_intervals1[core_name])
            base_time2 = compute_sum(time_intervals2[core_name])
            ratio1 = overlap_time / base_time1
            ratio2 = overlap_time / base_time2
            if core_name not in overlap_ratios:
                overlap_ratios[core_name] = {}
            overlap_ratios[core_name][key1] = ratio1
            overlap_ratios[core_name][key2] = ratio2
    return overlap_ratios


def compute_avg_overlap_ratio(summary):
    avg_summary = {}
    collection = {}
    for core_name in summary:
        for key in summary[core_name]:
            if key not in collection:
                collection[key] = []
            collection[key].append(summary[core_name][key])
    for key in collection:
        avg_summary[key] = sum(collection[key])/len(collection[key])
    return avg_summary


def get_flow_categories(dir_path):
    flow_path = os.path.join(dir_path, FILE_NAMES["flow"])
    flows = utils.read_json(flow_path)
    return flows["categories"]


def collect_special_event_time(data):
    if "ts" not in data or "dur" not in data or "cname" not in data:
        return
    if data["tid"] not in SWIM_LANE_INFO and data["cname"] != SWIM_LANE_INFO[data["tid"]]:
        return
    if data["dur"] == 0:
        return
    if data["name"] not in SPECIAL_EVENTS:
        return
    if data["tid"] not in SPECIAL_EVENT_TIME_INTERVAL:
        SPECIAL_EVENT_TIME_INTERVAL[data["tid"]] = {}
    if data["pid"] not in SPECIAL_EVENT_TIME_INTERVAL[data["tid"]]:
        SPECIAL_EVENT_TIME_INTERVAL[data["tid"]][data["pid"]] = {}
    if data["name"] not in SPECIAL_EVENT_TIME_INTERVAL[data["tid"]][data["pid"]]:
        SPECIAL_EVENT_TIME_INTERVAL[data["tid"]][data["pid"]][data["name"]] = []
    SPECIAL_EVENT_TIME_INTERVAL[data["tid"]][data["pid"]][data["name"]].append(
        [data["ts"], data["ts"] + data["dur"]]
    )


def dump_special_event_ratio(summary):
    text = "[Special Event Distribution Over Each Core]\n"
    transform = {}
    for key in summary:
        for key2 in summary[key]:
            if key2 not in transform:
                transform[key2] = {}
            if key not in transform[key2]:
                transform[key2][key] = {}
            for event_name in summary[key][key2]:
                transform[key2][key][event_name] = summary[key][key2][event_name]
    for key in sorted(transform.keys()):
        text += f"  [Pipe Distribution Over {key}]\n"
        for key2 in sorted(transform[key].keys()):
            for event_name in transform[key][key2]:
                text += f"    %({event_name} in {key2}): {transform[key][key2][event_name]:.2%}\n"
        text += "\n"
    text += "\n"
    return text


def compute_special_event_ratio():
    """
        summary格式：
        {
            "swim_lane_name": {
                "core_name": {
                    "event_name": xxx
                }
            }
        }
        """
    summary = {}
    for swim_lane_name in SPECIAL_EVENT_TIME_INTERVAL:
        summary[swim_lane_name] = {}
        for core_name in SPECIAL_EVENT_TIME_INTERVAL[swim_lane_name]:
            summary[swim_lane_name][core_name] = {}
            for event_name in SPECIAL_EVENT_TIME_INTERVAL[swim_lane_name][core_name]:
                non_idle_time = 0
                for item in SPECIAL_EVENT_TIME_INTERVAL[swim_lane_name][core_name][event_name]:
                    non_idle_time += item[1] - item[0]
                summary[swim_lane_name][core_name][event_name] = non_idle_time / SIMULATOR_RUNTIME[core_name]
    return summary


def compute_avg_special_event_ratio(summary):
    avg_summary = {}
    for swim_lane_name in summary:
        collection = {}
        for core_name in summary[swim_lane_name]:
            for event_name in summary[swim_lane_name][core_name]:
                if event_name not in collection:
                    collection[event_name] = []
                collection[event_name].append(summary[swim_lane_name][core_name][event_name])
        avg_summary[swim_lane_name] = {}
        for event_name in collection:
            avg_summary[swim_lane_name][event_name] = sum(collection[event_name]) / len(collection[event_name])
    return avg_summary


def get_simulator_runtime(data):
    if "ts" not in data or "pid" not in data:
        return
    end_time = data["ts"]
    if "dur" in data:
        end_time += data["dur"]
    if data["pid"] not in SIMULATOR_RUNTIME:
        SIMULATOR_RUNTIME[data["pid"]] = 0
    if end_time > SIMULATOR_RUNTIME[data["pid"]]:
        SIMULATOR_RUNTIME[data["pid"]] = end_time


def dump_avg_simulator_runtime():
    text = f"[Simulator Runtime]\n"
    avg = sum(SIMULATOR_RUNTIME.values()) / len(SIMULATOR_RUNTIME)
    text += f"  AVG(runtime): {avg:.4}\n"
    max_value = max(SIMULATOR_RUNTIME.values())
    text += f"  MAX(runtime): {max_value:.4}\n"
    min_value = min(SIMULATOR_RUNTIME.values())
    text += f"  MIN(runtime): {min_value:.4}\n"
    max_diff = max_value - min_value
    text += f"  MAX_DIFF(runtime): {max_diff:.4}\n"
    ratio = max_diff / max_value
    text += f"  %(MAX_DIFF/MAX): {ratio:.2%}\n"
    text += "\n"
    return text


def dump_simulator_runtime():
    text = "[Simulator Runtime Over Each Core]\n"
    for core_name in SIMULATOR_RUNTIME:
        text += f"  Runtime({core_name}): {SIMULATOR_RUNTIME[core_name]:.4}\n"
    text += "\n"
    return text


def dump_avg_non_idle_ratio(avg_non_idle_ratio, pipe_result=None):
    """Modified: adds instr/cycles columns from API_INSTR alongside dur% from trace."""
    if pipe_result:
        text = f"[Pipe Distribution]  instr count / instr% / cycles / cycles% / dur%\n"
        text += f"  Total instr: {pipe_result['total_instr']}  |  Total cycles: {pipe_result['total_cycles']}\n"
        for pname in sorted(pipe_result["pipes"].keys()):
            d = pipe_result["pipes"][pname]
            dur_pct = avg_non_idle_ratio.get(pname, 0)
            text += f"  {pname:12s}  instr={d['instr']:>7d} ({d['instr_pct']:>5.1f}%)  " \
                    f"cycles={d['cycles']:>10d} ({d['cycles_pct']:>5.1f}%)  " \
                    f"dur={dur_pct:>6.2%}\n"
    else:
        text = f"[Pipe Distribution]\n"
        for swim_lane_name in avg_non_idle_ratio:
            text += f"  %({swim_lane_name}): {avg_non_idle_ratio[swim_lane_name]:.2%}\n"
    text += "\n"
    return text


def dump_avg_special_event_ratio(avg_special_event_ratio):
    text = f"[Special Event Distribution]\n"
    for swim_lane_name in avg_special_event_ratio:
        for event_name in avg_special_event_ratio[swim_lane_name]:
            text += f"  %({event_name} in {swim_lane_name}): {avg_special_event_ratio[swim_lane_name][event_name]:.2%}\n"
    text += "\n"
    return text


def dump_avg_overlap_ratio(avg_overlap_ratio):
    text = f"[Pipe Overlap Ratio]\n"
    for key in avg_overlap_ratio:
        text += f"  %({key}): {avg_overlap_ratio[key]:.2%}\n"
    text += "\n"
    return text

def dump_flows(flows_data):
    """Modified: shows detailed pipeline flows with count/avg_delta/min/max instead of just category names."""
    if flows_data is None:
        return "[Pipeline Flows]  No flow data available\n\n"
    text = f"[Pipeline Flows]  category / count / avg_delta(ns) / min / max\n"
    for cf in flows_data.get("categoryFlows", []):
        cat = cf["category"]
        flist = cf["flows"]
        if flist:
            deltas = [f["to"]["ts"] - f["from"]["ts"] for f in flist]
            avg_delta = round(sum(deltas) / len(deltas), 1)
            min_delta = round(min(deltas), 1)
            max_delta = round(max(deltas), 1)
            text += f"  {cat:24s}  count={len(flist):>6d}  " \
                    f"avg={avg_delta:>8.1f}ns  " \
                    f"min={min_delta:>7.1f}ns  max={max_delta:>7.1f}ns\n"
    text += "\n"
    return text


def dump_key_ratios(ratios):
    text = "[Key Ratios]\n"
    for k, v in ratios.items():
        text += f"  {k} = {v}\n"
    text += "\n"
    return text


def dump_vector_unit(vector):
    text = "[VECTOR Unit]\n"
    text += f"  Instr count: {vector['instr_count']}\n"
    text += f"  UB Read Conflict:  {vector['ub_read_conflict']}\n"
    text += f"  UB Write Conflict: {vector['ub_write_conflict']}\n"
    text += f"  UB Conflict Total: {vector['ub_conflict_total']}\n"
    text += f"  Utilization avg/min/max = {vector['util_pct_avg']}% / " \
            f"{vector['util_pct_min']}% / {vector['util_pct_max']}%  " \
            f"(samples={vector['util_sample_count']})\n"
    if vector["top_conflicts"]:
        text += "  Top-conflict instrs:\n"
        for src, urc, uwc, util in vector["top_conflicts"][:5]:
            text += f"    RC={urc:>5d} WC={uwc:>5d} U={util:>5.1f}% | {src}\n"
    if vector["top_sources"]:
        text += "  Top instr types:\n"
        for src, cnt in list(vector["top_sources"].items())[:8]:
            text += f"    {cnt:>5d}: {src}\n"
    text += "\n"
    return text


def dump_mte2_data_transport(mte2):
    text = "[MTE2 Data Transport]\n"
    text += f"  Instr count: {mte2['instr_count']}  |  Data movers: {mte2['data_mover_count']}  " \
            f"|  Flow control: {mte2['control_op_count']}\n"
    text += f"  ProcessBytes / data mover:  avg={mte2['pb_data_mover_avg']:.0f}B  " \
            f"min={mte2['pb_data_mover_min']}B  max={mte2['pb_data_mover_max']}B\n"
    text += f"  ProcessBytes / all MTE2:    avg={mte2['pb_all_avg']:.0f}B  " \
            f"max={mte2['pb_all_max']}B\n"
    if mte2["top_sources"]:
        text += "  Top instr types:\n"
        for src, cnt in list(mte2["top_sources"].items())[:5]:
            text += f"    {cnt:>5d}: {src}\n"
    text += "\n"
    return text


def dump_cube_mma(cube):
    if not cube.get("present"):
        return ""
    text = "[CUBE/MMA]\n"
    text += f"  Instr count: {cube['instr_count']}  |  MMAD: {cube['mmad_count']}  " \
            f"|  Cycles: {cube['total_cycles']}\n"
    text += "\n"
    return text


def dump_wait_bar(flowctrl):
    text = "[WAIT_FLAG / BAR Sync]  Totals across all pipes\n"
    text += f"  WAIT_FLAG total: {flowctrl['wait_total']}  |  BAR total: {flowctrl['bar_total']}\n"
    if flowctrl["wait_by_pipe"]:
        text += f"  WAIT_FLAG by pipe: {flowctrl['wait_by_pipe']}\n"
    if flowctrl["bar_by_pipe"]:
        text += f"  BAR by pipe: {flowctrl['bar_by_pipe']}\n"
    text += "\n"
    return text


def dump_scalar_instr_types(scalar):
    text = "[SCALAR Instr Types]\n"
    text += f"  Instr count: {scalar['instr_count']}\n"
    if scalar["top_sources"]:
        for src, cnt in list(scalar["top_sources"].items())[:12]:
            text += f"    {cnt:>5d}: {src}\n"
    text += "\n"
    return text


def dump_trace_events(trace):
    text = "[TRACE Events]\n"
    text += f"  Total events: {trace['total_events']}  |  Complete events(ph=X): {trace['complete_events']}\n"
    if trace["top_events"]:
        text += "  Top-20 event names:\n"
        for name, cnt in trace["top_events"][:20]:
            text += f"    {cnt:>8d}: {name}\n"
    text += f"  Arithmetic events (SIGNEXT+ADD+MUL+DIV+SUB+MADD): {trace['arithmetic_total']} " \
            f"/ {trace['complete_events']} = {trace['arithmetic_pct_of_complete']}%\n"
    text += f"  Arithmetic breakdown: {trace['arithmetic']}\n"
    text += "\n"
    return text

def GenReport(dir_path, isTimeOut=False, max_source_line = 10):
    # Load instruction data
    instr_path = os.path.join(dir_path, FILE_NAMES["instr"])
    instructions = []
    cores = []
    if os.path.exists(instr_path):
        instr_data = utils.read_json(instr_path)
        instructions = instr_data.get("Instructions", [])
        cores = instr_data.get("Cores", [])

    trace_path = os.path.join(dir_path, FILE_NAMES["trace"])
    traces = utils.read_json(trace_path)
    for item in traces["traceEvents"]:
        collect_non_idle_time(item)
        collect_special_event_time(item)
        get_simulator_runtime(item)
    for swim_lane_name in NON_IDLE_TIME_INTERVAL:
        for core_name in NON_IDLE_TIME_INTERVAL[swim_lane_name]:
            NON_IDLE_TIME_INTERVAL[swim_lane_name][core_name] = collapse_time_interval(NON_IDLE_TIME_INTERVAL[swim_lane_name][core_name])
    for swim_lane_name in SPECIAL_EVENT_TIME_INTERVAL:
        for core_name in SPECIAL_EVENT_TIME_INTERVAL[swim_lane_name]:
            for event_name in SPECIAL_EVENT_TIME_INTERVAL[swim_lane_name][core_name]:
                SPECIAL_EVENT_TIME_INTERVAL[swim_lane_name][core_name][event_name] = collapse_time_interval(SPECIAL_EVENT_TIME_INTERVAL[swim_lane_name][core_name][event_name])
    non_idle_ratio = compute_non_idle_ratio()
    avg_non_idle_ratio = compute_avg_non_idle_ratio(non_idle_ratio)
    special_event_ratio = compute_special_event_ratio()
    avg_special_event_ratio = compute_avg_special_event_ratio(special_event_ratio)
    flow_path = os.path.join(dir_path, FILE_NAMES["flow"])
    flows_data = utils.read_json(flow_path) if os.path.exists(flow_path) else None
    overlap_ratio = compute_overlap_ratio()
    avg_overlap_ratio = compute_avg_overlap_ratio(overlap_ratio)

    pipe_result = analyze_pipe(instructions)
    ratios = analyze_ratios(pipe_result)
    vector = analyze_vector(instructions)
    mte2 = analyze_mte2(instructions)
    cube = analyze_cube(instructions)
    flowctrl = analyze_flowctrl(instructions)
    scalar = analyze_scalar(instructions)
    trace = analyze_trace(traces["traceEvents"])

    # Per-core pipe data (from API_INSTR)
    per_core_pipe = compute_pipe_per_core(instructions, cores)

    # Build report: aggregate section (top) + per-core section (bottom)
    text = ""
    # Header
    label = os.path.basename(os.path.dirname(dir_path.rstrip("/"))) or dir_path
    text += "=" * 72 + "\n"
    text += f"Kernel : {label}\n"
    text += f"Data   : {dir_path}\n"
    text += "=" * 72 + "\n"
    # Overall section
    text += "Overall\n"
    text += "=" * 72 + "\n"
    text += dump_avg_non_idle_ratio(avg_non_idle_ratio, pipe_result)
    text += dump_key_ratios(ratios)
    text += dump_vector_unit(vector)
    text += dump_mte2_data_transport(mte2)
    text += dump_cube_mma(cube)
    text += dump_wait_bar(flowctrl)
    text += dump_flows(flows_data)
    text += dump_scalar_instr_types(scalar)
    text += dump_trace_events(trace)
    text += dump_avg_special_event_ratio(avg_special_event_ratio)
    if not isTimeOut:
        text += dump_avg_simulator_runtime()
    text += dump_avg_overlap_ratio(avg_overlap_ratio)
    text += extract_api_instr.extract_source_features(dir_path, max_source_line)
    text += dump_compute_non_idle_ratio(non_idle_ratio)
    # Per-core section
    text += "=" * 72 + "\n"
    text += "Per-Core Detail\n"
    text += "=" * 72 + "\n"
    text += dump_compute_non_idle_ratio(non_idle_ratio, per_core_pipe)
    text += dump_special_event_ratio(special_event_ratio)
    if not isTimeOut:
        text += dump_simulator_runtime()
    text += dump_overlap_ratio(overlap_ratio)
    utils.write_txt(os.path.join(dir_path, "report.txt"), text)
    return text
