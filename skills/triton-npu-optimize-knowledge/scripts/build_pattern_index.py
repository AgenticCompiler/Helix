from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_pattern_catalog():
    script_path = Path(__file__).resolve()
    catalog_path = script_path.with_name("pattern_catalog.py")
    module_name = f"{script_path.parent.parent.name.replace('-', '_')}_pattern_catalog"
    spec = importlib.util.spec_from_file_location(module_name, catalog_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load pattern catalog: {catalog_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_PATTERN_CATALOG = _load_pattern_catalog()
PatternCard = _PATTERN_CATALOG.PatternCard
parse_pattern_card = _PATTERN_CATALOG.parse_pattern_card
list_high_priority_pattern_cards = _PATTERN_CATALOG.list_high_priority_pattern_cards
build_high_priority_reminder_lines = _PATTERN_CATALOG.build_high_priority_reminder_lines
build_index_text = _PATTERN_CATALOG.build_index_text
main = _PATTERN_CATALOG.main


if __name__ == "__main__":
    raise SystemExit(main())
