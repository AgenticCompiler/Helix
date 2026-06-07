"""
从 dataType_4_API_INSTR.json 中提取 "AscendC Inner Code"、"Source"、"Pipe" 的一一映射关系
"""
import utils
import os
import math

SOURCE_FILENAME = "dataType_4_API_INSTR.json"

AVG_KEYS = {"Cycles": "sum", "GPR Count": "max",
            "Instructions Executed": "sum", "Process Bytes": "sum",
            "UB Read Conflict": "sum", "UB Write Conflict": "sum",
            "Vector Utilization Percentage": "sum"}
DUMP_KEYS = ["#Source", ] + sorted(list(AVG_KEYS.keys()))
PIPE_KEY = "Pipe Distribution Over Cycles"
MAX_SOURCE = 10

def get_location(ascendc):
    # e.g., /mnt/data01/pandaoxin/bzhan/triton-dataset/DLBlas_origin/NPUKernelBench_level_1_2_triton_0519/split_3/2_SwiGLU/opt_triton_2_SwiGLU.py:15
    if len(ascendc) == 0:
        return {}
    tmp = ascendc.split(":")
    if len(tmp) <= 1:
        return {}
    try:
        return {"path": ":".join(tmp[:-1]), "LineNo": int(tmp[-1])}
    except ValueError:
        print(f"WARN: Failed to extract location from ascendc -> {ascendc}")
    return {}

def avg_data(data):
    filtered = []
    for item in data:
        if item < 0:
            continue
        filtered.append(item)
    if len(filtered) == 0:
        return -1
    return sum(filtered) / len(filtered)

def count_data(items, key, flag):
    filtered = [x[key] for x in items if x[key] > 0]
    if flag == "sum":
        return sum(filtered)
    elif flag == "max":
        if len(filtered) == 0:
            return 0
        else:
            return max(filtered)
    raise Exception(f"ERROR: Unknown flag -> {key} : {flag}")

def get_pipe_ration_given_cycle(items):
    summary = {}
    for item in items:
        if item["Cycles"] < 0:
            continue
        if item["Pipe"] not in summary:
            summary[item["Pipe"]] = 0
        summary[item["Pipe"]] += item["Cycles"]
    sum_data = sum(summary.values())
    ratios = {}
    if sum_data == 0:
        return ratios
    for key in summary.keys():
        ratios[key] = summary[key] / sum_data
    return ratios

def dump_source(summary):
    text = ""
    for key in DUMP_KEYS:
        text += f"    {key}: {summary[key]}\n"
    return text

def dump_pipe_distribution(summary):
    if len(summary) == 0:
        return ""
    text = f"    [{PIPE_KEY}] "
    for key in sorted(summary.keys()):
        text += f"{key}: {summary[key]:.2%}, "
    text = text[:-2] + "\n"
    return text


def dump_source_info(summary):
    #cycles -> code
    cycles_info = {}
    for filename in summary.keys():
        for line_no in summary[filename].keys():
            if "Cycles" not in summary[filename][line_no] or summary[filename][line_no]["Cycles"] == "NA":
                continue
            cycles_info[summary[filename][line_no]["Cycles"]] = [filename, line_no]
    if len(cycles_info) == 0:
        return ""
    cycles = sorted(cycles_info.keys(), reverse=True)[:MAX_SOURCE]
    text = "[Source Code Info]\n"
    for cycle in cycles:
        text += f"  Source Code: File: {cycles_info[cycle][0]}, LineNo: {cycles_info[cycle][1]}\n"
        text += dump_source(summary[cycles_info[cycle][0]][cycles_info[cycle][1]])
        text += dump_pipe_distribution(summary[cycles_info[cycle][0]][cycles_info[cycle][1]][PIPE_KEY])
        text += "\n"
    return text


def extract_mapping(input_path: str):
    data = utils.read_json(input_path)
    instructions = data.get("Instructions", [])
    mapping = {}

    for instr in instructions:
        ascendc = instr.get("AscendC Inner Code", "")
        location = get_location(ascendc)
        if len(location) == 0:
            continue
        path = location.get("path", "")
        if path not in mapping:
            mapping[path] = {}
        line_no = location.get("LineNo", "")
        if line_no not in mapping[path]:
            mapping[path][line_no] = []
        item = {
            "Source": instr.get("Source", ""),
            "Pipe": instr.get("Pipe", ""),
        }
        for key in AVG_KEYS:
            item[key] = avg_data(instr.get(key, ""))
        mapping[path][line_no].append(item)

    summary = {}
    for path in mapping.keys():
        summary[path] = {}
        for line_no in mapping[path].keys():
            summary[path][line_no] = {
                "#Source": len(mapping[path][line_no]),
                PIPE_KEY: get_pipe_ration_given_cycle(mapping[path][line_no]),
            }
            for key in AVG_KEYS:
                value = count_data(mapping[path][line_no], key, AVG_KEYS[key])
                if value > 0:
                    summary[path][line_no][key] = int(math.ceil(value))
                else:
                    summary[path][line_no][key] = "NA"
    return summary


def extract_source_features(dir_path, max_source_line):
    global MAX_SOURCE
    MAX_SOURCE = max_source_line
    source_file = os.path.join(dir_path, SOURCE_FILENAME)
    if not os.path.exists(source_file):
        return ""
    summary = extract_mapping(source_file)
    text = dump_source_info(summary)
    return text

