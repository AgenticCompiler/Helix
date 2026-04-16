import json
from typing import Any, Optional, Iterable


def _format_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _format_markdown_table(rows: Iterable[Iterable[Any]], headers: Iterable[Any]) -> str:
    header_cells = [_format_markdown_cell(cell) for cell in headers]
    body_rows = [[_format_markdown_cell(cell) for cell in row] for row in rows]
    column_count = max(
        [len(header_cells)] + [len(row) for row in body_rows],
        default=len(header_cells),
    )

    if len(header_cells) < column_count:
        header_cells.extend([""] * (column_count - len(header_cells)))

    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(["---"] * column_count) + " |",
    ]
    for row in body_rows:
        if len(row) < column_count:
            row = row + [""] * (column_count - len(row))
        lines.append("| " + " | ".join(row[:column_count]) + " |")
    return "\n".join(lines)


class BinaryJsonExtractor:
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
        self.marker = b'ZZ{'

    def extract_json_blocks(self, filename: str) -> list[dict[str, Any]]:
        """
        Extract all JSON blocks from binary file by searching for b'ZZ{' markers.

        Args:
            filename: Path to binary file

        Returns:
            List of parsed JSON dictionaries
        """
        json_blocks = []

        with open(filename, 'rb') as file:
            data = file.read()

        # Find all positions of 'ZZ{' markers
        marker_positions = self._find_markers(data)

        # Extract JSON from each marker position
        for pos in marker_positions:
            json_bytes = self._extract_json_bytes(data, pos + 2)  # Skip 'ZZ'
            if json_bytes:
                try:
                    json_str = json_bytes.decode(self.encoding)
                    json_obj = json.loads(json_str)
                    json_blocks.append(json_obj)
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    print(f"Warning: Failed to parse JSON at position {pos}: {e}")
                    # Try to fix common encoding issues
                    try:
                        json_str = json_bytes.decode(self.encoding, errors='replace')
                        json_obj = json.loads(json_str)
                        json_blocks.append(json_obj)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue

        return json_blocks

    def _find_markers(self, data: bytes) -> list[int]:
        """
        Find all positions where b'ZZ{' occurs in the binary data.

        Args:
            data: Binary data to search

        Returns:
            List of byte positions where marker starts
        """
        positions = []
        pos = 0
        marker_len = len(self.marker)
        data_len = len(data)

        while pos < data_len - marker_len + 1:
            # Find next occurrence of 'Z'
            found = data.find(b'Z', pos)
            if found == -1:
                break

            # Check if we have 'ZZ{' starting at this position
            if (found <= data_len - marker_len and
                data[found:found+marker_len] == self.marker):
                positions.append(found)
                pos = found + 1  # Continue searching
            else:
                pos = found + 1  # Move past this 'Z'

        return positions

    def _extract_json_bytes(self, data: bytes, start_pos: int) -> Optional[bytes]:
        """
        Extract JSON bytes starting from a position where '{' is expected.

        Args:
            data: Binary data
            start_pos: Starting position (should point to '{')

        Returns:
            JSON bytes (including the opening '{') or None if invalid
        """
        # Verify we're starting with '{'
        if start_pos >= len(data) or data[start_pos:start_pos+1] != b'{':
            return None

        brace_count = 0
        in_string = False
        escape_next = False
        json_start = start_pos

        for i in range(start_pos, len(data)):
            byte = data[i:i+1]

            # Handle string literals
            if not escape_next and byte == b'"':
                in_string = not in_string
            elif in_string and byte == b'\\':
                escape_next = not escape_next
                continue
            elif escape_next:
                escape_next = False
                continue

            # Count braces if not in string
            if not in_string:
                if byte == b'{':
                    brace_count += 1
                elif byte == b'}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found matching closing brace
                        return data[json_start:i+1]

        return None  # Unbalanced braces

    def extract_with_positions(self, filename: str) -> list[dict]:
        """
        Extract JSON blocks with their positions in the file.

        Returns:
            List of dictionaries containing JSON and metadata
        """
        results = []

        with open(filename, 'rb') as file:
            data = file.read()

        marker_positions = self._find_markers(data)

        for marker_pos in marker_positions:
            json_start = marker_pos + 2  # Skip 'ZZ'
            json_bytes = self._extract_json_bytes(data, json_start)

            if json_bytes:
                try:
                    json_str = json_bytes.decode(self.encoding)
                    json_obj = json.loads(json_str)

                    results.append({
                        'json': json_obj,
                        'marker_position': marker_pos,
                        'json_start': json_start,
                        'json_end': json_start + len(json_bytes),
                        'total_size': len(json_bytes) + 2,  # Including 'ZZ'
                        'json_bytes': json_bytes  # Raw bytes for debugging
                    })
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue

        return results

    def extract_and_output(self, filename: str):
        """
        Extract JSON blocks from file, and write the results to new files with
        suffix _part{i}.

        """
        results = self.extract_with_positions(filename)
        for i, result in enumerate(results, 1):
            with open(filename[:-4] + f"part_{i}.json", 'w', encoding='utf-8') as output:
                output.write(json.dumps(result['json'], indent=4))


class BaseInfo:
    def __init__(self, *, name: str, duration: float, op_type: str, block_dim: int,
                 head_name: Iterable[str], block_detail: Iterable[tuple[str]]):
        self.name = name
        self.duration = duration
        self.op_type = op_type
        self.block_dim = block_dim
        self.head_name = tuple(head_name)
        self.block_detail = tuple(block_detail)

    def print_info(self):
        res = ""
        res += f"**Name:** {self.name}\n\n"
        res += f"**Duration:** {self.duration}\n\n"
        res += f"**Op Type:** {self.op_type}\n\n"
        res += f"**Block Dim:** {self.block_dim}\n\n"
        if self.op_type == "vector":
            res += "#### Block Detail (at most 20 printed)\n\n"
            res += _format_markdown_table(self.block_detail[:20], self.head_name) + "\n"
        elif self.op_type == "mix":
            res += "#### Mix Block Detail (at most 20 printed)\n\n"
            res += _format_markdown_table(self.block_detail[:20], self.head_name) + "\n"
        elif self.op_type == "cube":
            res += "#### Block Detail (at most 20 printed)\n\n"
            res += _format_markdown_table(self.block_detail[:20], self.head_name) + "\n"
        else:
            raise AssertionError(f"op_type = {self.op_type}")

        return res


class WorkloadAnalysis:
    def __init__(self, pipe_utilization: Iterable[tuple[str]], details: dict[int, list[tuple[str]]]):
        self.pipe_utilization = tuple(pipe_utilization)
        self.details = details

    def print_info(self, *, block_id: int):
        res = ""
        res += "#### Pipe utilization (at most 20 printed)\n\n"
        headers = ["Block ID", "Name", "Utilization (%)"]
        res += _format_markdown_table(self.pipe_utilization[:20], headers) + "\n\n"
        if block_id in self.details:
            res += f"#### Details for block {block_id} (at most 20 printed)\n\n"
            headers = ["VECTOR", "Instructions", "Duration (us)", "Data volume (byte)"]
            res += _format_markdown_table(self.details[block_id][:20], headers) + "\n"
        return res


data_path_description = {
    0: "GM -> L2 Cache",
    1: "L2 Cache -> GM",
    2: "L2 Cache -> L1",
    3: "L1 -> L2 Cache",
    4: "L1 -> L0A",
    5: "L1 -> L0B",
    6: "L0A -> Cube",
    7: "L0B -> Cube",
    8: "Cube -> L0C",
    9: "L0C -> Cube",
    10: "L0C -> L2 Cache",
    11: "L0C -> L1",
    12: "L2 Cache -> UB (Vector0)",
    13: "UB -> L2 Cache (Vector0)",
    14: "UB -> Vector (Vector0)",
    15: "Vector -> UB (Vector 0)",
    16: "L2 Cache -> UB (Vector1)",
    17: "UB -> L2 Cache (Vector1)",
    18: "UB -> Vector (Vector1)",
    19: "Vector -> UB (Vector1)",
}


class CoreMemoryMap:
    def __init__(self, advice: Iterable[int], data_paths: Iterable[tuple[str]],
                 l2_cache: Optional[dict], vector: Optional[dict], vector1: Optional[dict],
                 cube: Optional[dict]):
        self.advice = tuple(advice)
        self.data_paths = tuple(data_paths)
        self.l2_cache = l2_cache
        self.vector = vector
        self.vector1 = vector1
        self.cube = cube

    def print_info(self):
        res = ""
        if self.advice:
            res += "#### Advice\n\n"
            for advice in self.advice:
                res += advice + "\n\n"
        if self.l2_cache:
            res += f"**L2 Cache Hit Rate:** {float(self.l2_cache['hit_ratio']):.2f}%\n\n"
        if self.vector:
            res += f"**Vector Ratio:** {float(self.vector['ratio']):.2f}%\n\n"
        if self.vector1:
            res += f"**Vector1 Ratio:** {float(self.vector1['ratio']):.2f}%\n\n"
        if self.cube:
            res += f"**Cube Ratio:** {float(self.cube['ratio']):.2f}%\n\n"
        res += "#### Data paths\n\n"
        headers = ["Path", "Bandwidth (GB/s)", "Request"]
        res += _format_markdown_table(self.data_paths, headers) + "\n\n"
        return res


class MemoryWorkloadTable:
    def __init__(self, advice: Iterable[str], table_detail: Iterable[dict]):
        self.advice = tuple(advice)
        self.table_detail = tuple(table_detail)

    def print_info(self):
        res = ""
        if self.advice:
            res += "#### Advice\n\n"
            for advice in self.advice:
                res += advice + "\n\n"
        for table in self.table_detail:
            headers = table['header_name']
            headers[0] = table['table_name']
            data = []
            for row in table['row']:
                data.append([row['name']] + row['value'])
            res += f"#### {table['table_name']}\n\n"
            res += _format_markdown_table(data, headers) + "\n\n"
        return res


class MemoryWorkloadAnalysis:
    def __init__(self, core_memory_maps: dict[int, CoreMemoryMap],
                 workload_tables: dict[int, MemoryWorkloadTable]):
        self.core_memory_maps = core_memory_maps
        self.workload_tables = workload_tables

    def print_info(self, *, block_id: int):
        res = ""
        if block_id in self.core_memory_maps:
            res += f"### Core memory map for block {block_id}\n\n"
            res += self.core_memory_maps[block_id].print_info()
        if block_id in self.workload_tables:
            res += f"### Memory workload table for block {block_id}\n\n"
            res += self.workload_tables[block_id].print_info()
        return res


class AllInfo:
    def __init__(self, base_info: BaseInfo, workload_analysis: WorkloadAnalysis,
                 memory_workload_analysis: MemoryWorkloadAnalysis):
        self.base_info = base_info
        self.workload_analysis = workload_analysis
        self.memory_workload_analysis = memory_workload_analysis

    def print_info(self, *, block_id: int):
        res = ""
        res += "## Base Info\n\n"
        res += self.base_info.print_info() + "\n"
        res += "## Compute Workload Analysis\n\n"
        res += self.workload_analysis.print_info(block_id=block_id) + "\n"
        res += "## Memory Workload Analysis\n\n"
        res += self.memory_workload_analysis.print_info(block_id=block_id)
        return res


def get_info(results: dict) -> AllInfo:
    # Get base info
    result = results[0]['json']
    block_dim = int(result['block_dim']) if result['block_dim'] else 0
    op_type = result['op_type']
    block_detail = []
    if op_type == 'vector':
        head_name = result['block_detail']['head_name']
        for block in range(len(result['block_detail']['row'])):
            block_detail.append(tuple(result['block_detail']['row'][block]['value']))
    elif op_type == 'mix':
        head_name = result['mix_block_detail']['head_name']
        for block in range(len(result['mix_block_detail']['row'])):
            block_detail.append(tuple(result['mix_block_detail']['row'][block]['value']))
    elif op_type == 'cube':
        head_name = result['block_detail']['head_name']
        for block in range(len(result['block_detail']['row'])):
            block_detail.append(tuple(result['block_detail']['row'][block]['value']))
    else:
        raise AssertionError(f"op_type = {op_type}")

    base_info = BaseInfo(
        name=result['name'],
        duration=result['duration'],
        op_type=op_type,
        block_dim=block_dim,
        head_name=head_name,
        block_detail=block_detail
    )

    # Get pipe utilization
    result = results[1]['json']
    pipe_utilization = []
    for item in result['subblock_detail']:
        pipe_utilization.append((item['block_id'], item['name'], item['value']))

    # Get detailed info
    result = results[2]['json']
    details: dict[int, list[tuple[str]]] = dict()
    for item in result['subblock_detail']:
        block_id = int(item['block_id'])
        name = item['name']
        value = item['value']
        if block_id not in details:
            details[block_id] = list()
        if name == "Vector Wait":
            details[block_id].append((name, "", value, ""))
        elif name in ("Vector Compute Data Size", "Cube Compute Data Size"):
            details[block_id].append((name, "", "", value))
        else:
            details[block_id].append((name, value, "", ""))
    workload_analysis = WorkloadAnalysis(
        pipe_utilization=pipe_utilization,
        details=details
    )

    # Get path info
    result = results[3]['json']['core_memory_map']
    core_memory_maps = dict()
    for item in result:
        core_no = int(item['core_no'])
        data_paths = []
        for unit_item in item['memory_unit']:
            data_paths.append((data_path_description[unit_item['memory_path']], unit_item['bandwidth'], unit_item['request']))
            core_memory_maps[core_no] = CoreMemoryMap(
                advice=item['advice'],
                data_paths=data_paths,
                l2_cache=item.get('L2cache'),
                vector=item.get('Vector'),
                vector1=item.get('Vector1'),
                cube=item.get('Cube')
            )

    # Get workload tables
    result = results[4]['json']['table_per_block']
    workload_tables = dict()
    for item in result:
        block_id = int(item['block_id'])
        workload_tables[block_id] = MemoryWorkloadTable(
            advice=item['advice'],
            table_detail=item['table_detail']
        )

    memory_workload_analysis = MemoryWorkloadAnalysis(
        core_memory_maps=core_memory_maps,
        workload_tables=workload_tables
    )

    return AllInfo(base_info, workload_analysis, memory_workload_analysis)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse MindStudio Profiler binary output")
    parser.add_argument("filename", help="Path to the binary profiler output file")
    parser.add_argument("--block-id", type=int, default=0, help="Block ID to analyze (default: 0)")
    args = parser.parse_args()

    extractor = BinaryJsonExtractor()
    results = extractor.extract_with_positions(args.filename)
    info = get_info(results)
    print(info.print_info(block_id=args.block_id))
