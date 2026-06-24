# Optimize Knowledge Selection Design

## Goal

Add one `optimize` workflow option that selects which optimize knowledge library to stage before launching the agent, while preserving the staged skill name that the agent sees today.

This option must be supported by both `optimize` and `optimize-batch`.

## User-Visible Behavior

- Add `--optimize-knowledge {v1,v2}` to `optimize`.
- Add `--optimize-knowledge {v1,v2}` to `optimize-batch`.
- Default the option to `v1` so existing behavior does not change.
- Map `v1` to the repository skill directory `skills/triton/triton-npu-optimize-knowledge/`.
- Map `v2` to the repository skill directory `skills/triton/triton-npu-optimize-knowledge-v2/`.
- Keep the staged workspace skill name stable as `triton-npu-optimize-knowledge` for both versions.

## Design

### CLI Surface

The parser should treat optimize knowledge selection as an explicit enum instead of a boolean flag. This keeps the interface extensible and avoids coupling the contract to a single alternate library.

Both optimize commands should parse:

```text
--optimize-knowledge {v1,v2}
```

The parsed value should flow through `OptimizeRunOptions` as `optimize_knowledge: Literal["v1", "v2"]`, with a default of `v1`.

### Request Construction

`build_optimize_request()` should continue to stage the stable skill name `triton-npu-optimize-knowledge` through `staged_skill_names`, because optimize prompts and skill-relative references already rely on that name.

When the selected version is `v2`, request construction should also attach a staging override that tells the staging layer to copy:

- source: `triton-npu-optimize-knowledge-v2`
- target name in the workspace: `triton-npu-optimize-knowledge`

When the selected version is `v1`, no override is needed because the source and target names already match.

### Skill Staging

`SkillLinkManager` should gain a small, explicit aliasing capability for selected staged skills. The staging contract should stay narrow:

- callers still provide the stable target skill names through `skill_names`
- callers may optionally provide a mapping from staged target name to repository source directory name
- the staging layer copies from the mapped source directory when an override exists
- the staging layer still creates the target directory using the stable staged name

This keeps aliasing localized to staging and avoids spreading source-directory knowledge into optimize prompt or workflow logic.

### Failure Behavior

The implementation should fail explicitly when a requested source skill directory does not exist. It must not silently fall back from `v2` to `v1`.

Existing symlink and cleanup safety behavior in `SkillLinkManager` should remain unchanged.

## Testing

Add or update tests in these areas:

- `tests/test_cli.py`
  Verify `optimize` and `optimize-batch` both parse `--optimize-knowledge`, default it to `v1`, and pass `v2` through `optimize_run_options_from_args()`.
- `tests/test_optimize_runtime.py`
  Verify the default optimize request keeps the existing staged skill names and uses `v1`.
  Verify a `v2` request still stages `triton-npu-optimize-knowledge` as the target name while carrying the source override to `triton-npu-optimize-knowledge-v2`.
- `tests/test_skills.py`
  Verify staging can copy from an alternate source directory while preserving the staged target directory name.

## Scope Boundaries

- Do not rename either knowledge skill directory in the repository.
- Do not change optimize prompt wording that references `triton-npu-optimize-knowledge`.
- Do not change non-optimize commands.
- Do not introduce silent fallback between knowledge versions.
