# Diff Skills Update

## User-visible behavior

`triton-agent diff-skills-update -i <operators-root>` scans one level of operator
directories under the input root. Each operator directory may contain one or more
`opt_*.py` files. For every `opt_xxx.py`, the command looks for a sibling
`xxx.py`; pairs without either side are skipped with an explicit reason.

Each valid pair uses `xxx.py` as the pre-optimization baseline and `opt_xxx.py`
as the expected optimized answer. The command compares the pair, updates an
editable skills workspace, then runs a simulate agent from a local `simulate/`
directory that contains only the baseline source and staged skills. The simulate
agent is told which pattern names matched, but it is not given the answer file or
the diff. An analysis agent compares the generated candidate with `opt_xxx.py`.
If the candidate is not aligned, the analysis result drives another skills
update and simulate iteration until the pair aligns or `--max-iterations` is
reached.

## Paths

- Input root: CLI `-i/--input`.
- Skills workspace: `--skills-dir`, defaulting to `<operators-root>/skills`.
- Per-pair simulate workspace: `<operator-dir>/simulate/`.
- Per-pair report: `<operator-dir>/simulate/report.json`.
- Generated candidate: `<operator-dir>/simulate/generated_<stem>.py`.

When the skills workspace does not exist, it is seeded from the bundled
`skills/triton-npu-optimize-knowledge` skill. The command only edits the
workspace copy, never the bundled skill directory. Skills updates may revise
existing pattern cards or add new generic pattern cards when the diff exposes a
mechanism that is not covered yet.

If `--promote-converged-skills` is set, each pair that reaches `aligned` promotes
the editable `triton-npu-optimize-knowledge` workspace back over the bundled
`skills/triton-npu-optimize-knowledge` directory and rebuilds `pattern_index.md`
there. This option is off by default.

## Report Contract

`simulate/report.json` records the operator directory, baseline file, expected
file, matched pattern names, iteration results, and final status. A skipped pair
records `status: "skipped"` and a skip reason. A failed or incomplete pair
records enough agent output for a human to continue from the last iteration.
