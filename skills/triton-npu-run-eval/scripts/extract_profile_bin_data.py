#!/usr/bin/env python3
"""
Extract JSON data blocks from visualize_data.bin

Usage:
  python3 extract_bin_data.py <visualize_data.bin> [output_dir]
  python3 extract_bin_data.py <directory>          [output_dir]   # process all bin files found
  python3 extract_bin_data.py --flows <trace.json> [output_dir]   # extract flows only
"""

import glob
import json
import os
import struct
import sys
import utils
import advance_features

DATA_TYPE_NAMES = {
    1: "SOURCE",
    2: "TRACE",
    3: "API_FILE",
    4: "API_INSTR",
    5: "DETAILS_BASE_INFO",
    6: "DETAILS_COMPUTE_LOAD_GRAPH",
    7: "DETAILS_COMPUTE_LOAD_TABLE",
    8: "DETAILS_MEMORY_GRAPH",
    9: "DETAILS_MEMORY_TABLE",
    12: "DETAILS_INTER_CORE_LOAD_GRAPH",
    13: "DETAILS_ROOFLINE",
    14: "DISPLAY_CACHE",
}

HEADER_SIZE = 12  # dataSize(8) + dataType(1) + paddingLength(1) + instrVersion(1) + reserve(1)
FILE_PATH_LEN = 4096


def sanitize_filename(name, max_len=80):
    """Make a string safe for use as a filename component."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return safe[:max_len]


def extract_flows(trace_obj, output_dir):
    """Extract flow events (ph='s'/'t') from a Chrome Trace JSON and write flows.json.

    Flow events represent pipeline dependency arrows in the Timeline view.
    Each flow pair (s=start, t=target) shares the same `id` and `cat` (category).
    """
    events = trace_obj.get("traceEvents", [])
    flow_raw = [e for e in events if e.get("ph") in ("s", "t")]

    if not flow_raw:
        return 0

    # Group s/t events by id for pairing
    by_id = {}
    for e in flow_raw:
        fid = e.get("id", "")
        if fid not in by_id:
            by_id[fid] = {"s": None, "t": None}
        by_id[fid][e["ph"]] = e

    # Build paired flow items
    flows = []
    unpaired = 0
    for fid, pair in sorted(by_id.items()):
        s_event = pair.get("s")
        t_event = pair.get("t")
        if s_event and t_event:
            flows.append({
                "cat": s_event.get("cat", ""),
                "id": fid,
                "from": {"pid": s_event.get("pid", ""), "tid": s_event.get("tid", ""), "ts": s_event.get("ts", 0)},
                "to":   {"pid": t_event.get("pid", ""), "tid": t_event.get("tid", ""), "ts": t_event.get("ts", 0)},
            })
        else:
            unpaired += 1

    # Sort by from.ts, then id
    flows.sort(key=lambda f: (f["from"]["ts"], f["id"]))

    # Compute categories and summary
    categories = sorted(set(f["cat"] for f in flows))
    summary = {}
    for f in flows:
        summary[f["cat"]] = summary.get(f["cat"], 0) + 1

    # Group flows by category
    cat_flows_map = {cat: [] for cat in categories}
    for f in flows:
        cat_flows_map[f["cat"]].append(f)

    category_flows = []
    for cat in categories:
        cat_flows = cat_flows_map[cat]
        # Strip redundant 'cat' from each flow item (already in the group key)
        flow_items = []
        for f in cat_flows:
            item = {"id": f["id"], "from": f["from"], "to": f["to"]}
            flow_items.append(item)
        category_flows.append({
            "category": cat,
            "count": len(cat_flows),
            "flows": flow_items,
        })

    # Write output
    out_path = os.path.join(output_dir, "flows.json")
    utils.write_json(out_path, {"categories": categories, "summary": summary, "categoryFlows": category_flows})

    print(f"  [Flows] {len(flows)} flow pairs across {len(categories)} categories -> flows.json")
    for cat in categories:
        print(f"    {cat}: {summary[cat]}")
    if unpaired:
        print(f"    (unpaired s/t events: {unpaired})")

    return len(flows)


def extract_blocks(bin_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_size = os.path.getsize(bin_path)
    block_index = 0
    type_counters = {}  # track duplicate dataType indices for naming
    errors = []

    with open(bin_path, "rb") as f:
        while f.tell() < file_size:
            pos_before_header = f.tell()

            # Read header
            header = f.read(HEADER_SIZE)
            if len(header) < HEADER_SIZE:
                print(f"  [WARN] Incomplete header at offset {pos_before_header}, stopping.")
                break

            data_size = struct.unpack("<Q", header[0:8])[0]
            data_type = header[8]
            padding_length = header[9]
            instr_version = header[10]
            reserve = header[11]

            type_name = DATA_TYPE_NAMES.get(data_type, f"UNKNOWN_{data_type}")

            # Validate
            if data_size > file_size:
                print(f"  [ERROR] Block {block_index}: dataSize={data_size} exceeds file size={file_size}")
                break

            # Compute output filename with block index to avoid overwrites
            type_counter = type_counters.get(data_type, 0)
            type_counters[data_type] = type_counter + 1
            type_label = f"dataType_{data_type}_{type_name}"
            if type_counter > 0 or data_type == 1:  # always index for SOURCE (often multiple)
                suffix = f"_{type_counter}"
            else:
                suffix = ""

            if data_type == 1:  # SOURCE
                # SOURCE layout: [header 12B] [filePath 4096B] [sourceContent dataSize B] [padding paddingLength B]
                # dataSize does NOT include the 4096-byte filePath prefix
                file_path_bytes = f.read(FILE_PATH_LEN)
                source_file_path = file_path_bytes.decode("utf-8", errors="replace").strip("\x00")
                content_bytes = f.read(data_size - padding_length)
                if padding_length > 0:
                    f.read(padding_length)

                # Build a meaningful filename from the source file path
                src_basename = os.path.basename(source_file_path) if source_file_path else "unknown"
                safe_name = sanitize_filename(src_basename)
                out_name = f"{type_label}{suffix}_{safe_name}.txt"
                out_path = os.path.join(output_dir, out_name)

                # Write source content with filePath header
                with open(out_path, "w", encoding="utf-8", errors="replace") as out_f:
                    out_f.write(f"// Source File: {source_file_path}\n")
                    out_f.write(content_bytes.decode("utf-8", errors="replace"))

                print(f"  [Block {block_index}] {type_label}{suffix}: "
                      f"source={source_file_path[:60]}, content={len(content_bytes)}B "
                      f"-> {out_name}")

            else:
                # Non-SOURCE layout: [header 12B] [jsonContent dataSize B] [padding paddingLength B]
                content_bytes = f.read(data_size - padding_length)
                if padding_length > 0:
                    f.read(padding_length)

                out_name = f"{type_label}{suffix}.json"
                out_path = os.path.join(output_dir, out_name)

                try:
                    json_str = content_bytes.decode("utf-8")
                    json_obj = json.loads(json_str)
                    utils.write_json(out_path, json_obj)
                    # Summarize content
                    summary = ""
                    if isinstance(json_obj, dict):
                        keys = list(json_obj.keys())
                        summary = f"keys={keys[:5]}"
                        # Count traceEvents if present
                        if "traceEvents" in json_obj:
                            summary += f", events={len(json_obj['traceEvents'])}"
                        if "Instructions" in json_obj:
                            summary += f", instructions={len(json_obj['Instructions'])}"
                        if "Files" in json_obj:
                            summary += f", files={len(json_obj['Files'])}"
                    print(f"  [Block {block_index}] {type_label}{suffix}: "
                          f"{len(content_bytes)}B {summary} -> {out_name}")
                    # Extract flows from TRACE blocks
                    if data_type == 2:
                        extract_flows(json_obj, output_dir)
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    raw_name = f"{type_label}{suffix}.raw"
                    raw_path = os.path.join(output_dir, raw_name)
                    with open(raw_path, "wb") as out_f:
                        out_f.write(content_bytes)
                    errors.append(f"Block {block_index} {type_label}: JSON parse failed ({e})")
                    print(f"  [Block {block_index}] {type_label}{suffix}: "
                          f"{len(content_bytes)}B JSON PARSE FAILED -> {raw_name}")

            block_index += 1

    print(f"  Total blocks: {block_index}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors:
            print(f"    - {e}")

    return block_index, errors


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <visualize_data.bin | directory> [output_dir] [--isTimeOut]")
        print(f"       {sys.argv[0]} --flows <trace.json> [output_dir]")
        sys.exit(1)

    max_loc_num = 10

    isTimeOut = "--isTimeOut" in sys.argv
    if isTimeOut:
        sys.argv.remove("--isTimeOut")

    # --flows mode: extract flows from a standalone trace.json
    if sys.argv[1] == "--flows":
        if len(sys.argv) < 3:
            print(f"Usage: {sys.argv[0]} --flows <trace.json> [output_dir]")
            sys.exit(1)
        trace_path = sys.argv[2]
        if not os.path.isfile(trace_path):
            print(f"Error: file not found: {trace_path}")
            sys.exit(1)
        output_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(trace_path)
        os.makedirs(output_dir, exist_ok=True)
        trace_obj = utils.read_json(trace_path)
        count = extract_flows(trace_obj, output_dir)
        if count == 0:
            print("  No flow events found in trace.json")
        sys.exit(0)

    input_path = sys.argv[1]

    # Determine if input is a file or directory
    if os.path.isfile(input_path):
        bin_files = [input_path]
    elif os.path.isdir(input_path):
        bin_files = sorted(glob.glob(os.path.join(input_path, "**/visualize_data.bin"), recursive=True))
        if not bin_files:
            print(f"No visualize_data.bin found under {input_path}")
            sys.exit(1)
        print(f"Found {len(bin_files)} visualize_data.bin file(s)\n")
    else:
        print(f"Error: path not found: {input_path}")
        sys.exit(1)

    total_blocks = 0
    total_errors = []
    features_count = 0

    for bin_path in bin_files:
        short_path = os.path.relpath(bin_path)
        file_size = os.path.getsize(bin_path)
        print(f"=== {short_path} ({file_size:,} bytes) ===")

        # Default output dir: <bin_dir>/extracted_bin_data
        if len(sys.argv) > 2:
            output_dir = sys.argv[2]
        else:
            output_dir = os.path.join(os.path.dirname(bin_path), "extracted_bin_data")

        blocks, errors = extract_blocks(bin_path, output_dir)
        total_blocks += blocks
        total_errors.extend(errors)

        # Generate combined report (aggregate + per-core) after extraction
        try:
            report = advance_features.GenReport(output_dir, isTimeOut=isTimeOut, max_source_line = max_loc_num)
            if report:
                print(f"  -> report.txt written ({len(report.splitlines())} lines)")
                features_count += 1
        except Exception as e:
            print(f"  [WARN] report generation failed: {e}")

        print()

    print(f"{'='*60}")
    print(f"Summary: {len(bin_files)} file(s), {total_blocks} block(s) extracted, {len(total_errors)} error(s)")
    if features_count:
        print(f"         {features_count} report.txt generated")


if __name__ == "__main__":
    main()
