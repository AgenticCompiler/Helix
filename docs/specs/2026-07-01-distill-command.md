# Distill

## User-visible behavior

`helix distill -i <operators-root>` supports two explicit
input sources selected by `--source`.
The default code-agent backend is `opencode`.
The active operator language comes from `--lang` / `--language`, defaulting to
`triton`; supported values include `triton` and `tilelang`.

In the default `--source diff`, the command scans one level of operator
directories under the input root. Each operator directory may contain one or more
`opt_*.py` files. For every `opt_xxx.py`, the command looks for a sibling
`xxx.py`; pairs without either side are skipped with an explicit reason. This
source does not treat `learned_lessons.md` as an optimize workspace marker.

In `--source optimize`, the input is a completed optimize workspace or a
parent of completed optimize workspaces. The command only processes directories
that contain `learned_lessons.md`; operator directories without
`learned_lessons.md` are skipped instead of falling back to `opt_*.py` diff
discovery. For each optimize workspace, it uses `baseline/state.json` (with
fallback scanning under `baseline/`) to find the pre-optimization operator, uses
`opt-note.md`'s final best round or the latest `opt-round-N/` to find the
optimized operator, and gives the distill agent `learned_lessons.md`,
`opt-note.md`, and round `summary.md`/`attempts.md` context.

Each valid pair uses `xxx.py` as the pre-optimization baseline and `opt_xxx.py`
as the expected optimized answer. The command compares the pair, updates an
internal transient skills workspace, then runs a simulate agent from a local
`distill-simulator/` directory that contains only the baseline source and staged
skills.
The simulate agent is told which pattern names matched, but it is not given the
answer file or the diff. An analysis agent compares the generated candidate with
`opt_xxx.py`. If the candidate is not aligned, the analysis result drives
another skills update and simulate iteration until the pair aligns or
`--max-refine-rounds` is reached.

## Paths

- Input root: CLI `-i/--input`.
- Input source: `--source diff|optimize|git`, defaulting to `diff`.
- Internal skills workspace: transient
  `<operators-root>/.helix/distill-skills`, removed when the command
  returns.
- Output directory: `--output-dir`, defaulting to
  `<operators-root>/distill-output`. After the run completes, only pattern cards
  that changed during this run are copied into the output directory.
- Per-pair simulate workspace: `<operator-dir>/distill-simulator/`.
- Per-pair report: `<operator-dir>/distill-simulator/report.json`.
- Generated candidate: `<operator-dir>/distill-simulator/generated_<stem>.py`.

When the editable knowledge workspace does not exist, it is seeded from the
bundled `<language>-npu-optimize-knowledge` skill. For example, `--lang triton`
uses `triton-npu-optimize-knowledge`, while `--lang tilelang` uses
`tilelang-npu-optimize-knowledge`. The command only edits the workspace copy,
never the bundled skill directory. Skills updates may revise existing pattern
cards or add new generic pattern cards when the diff exposes a mechanism that is
not covered yet.

If `--promote-aligned` is set, each pair that reaches `aligned` promotes the
editable `<language>-npu-optimize-knowledge` workspace back over the matching
bundled skill directory and rebuilds `pattern_index.md` there. This option is off
by default.

## Report Contract

`distill-simulator/report.json` records the operator directory, baseline file,
expected file, matched pattern names, iteration results, and final status. A
skipped pair records `status: "skipped"` and a skip reason. A failed or
incomplete pair records enough agent output for a human to continue from the last
iteration.
