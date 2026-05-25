# Optimize Target Design

## Summary

- Add an explicit optimize target option, `--optimize-target kernel|operator`, to `optimize` and `optimize-batch`.
- Keep `kernel` as the default so existing optimize behavior does not change unless the user opts in.
- In `kernel` mode, keep the current contract that optimize work must focus on the Triton Ascend NPU kernel path itself.
- In `operator` mode, treat end-to-end operator latency as the optimization target and allow changes across the whole operator implementation, including wrapper logic, data movement, scheduling, pre/post-processing, and kernel code.
- In both modes, continue forbidding pure PyTorch rewrites that bypass the Triton Ascend NPU computation path.
- Tie optimize performance comparisons to target-aware `compare-perf` behavior instead of leaving metric-source selection implicit.

## Problem

The current optimize contract is intentionally narrow. The CLI, optimize prompt, and skill guidance all assume that the optimization target is the Triton kernel path itself. That works for kernel-centric tuning, but it does not let users explicitly ask for a broader operator-level latency objective.

When the real bottleneck is split across wrapper code, launch structure, data preparation, or kernel boundaries, the current contract is too restrictive:

- the CLI has no supported way to express the optimization target
- prompt guidance tells the agent to keep optimizing only the kernel path
- supervisor guidance audits rounds against kernel-only success criteria
- temporary workspace guidance does not explain whether the session should optimize the whole operator or only the kernel path

As a result, users cannot cleanly request an end-to-end operator optimization flow without relying on ad hoc prompt text, and the optimize system cannot consistently enforce or document that broader mode.

## Goals

- Give users a first-class CLI option to choose the optimize target.
- Preserve current behavior by default.
- Make `operator` mode explicit in optimize prompts, resume prompts, supervisor audits, and workspace guidance.
- Allow `operator` mode to optimize the full operator implementation rather than only the kernel body.
- Keep the Triton Ascend NPU path as a hard requirement in both modes.
- Keep the CLI thin by threading one explicit mode through existing optimize orchestration rather than inventing a separate command.

## Non-Goals

- Do not add a new optimize subcommand.
- Do not change baseline preparation, round directory layout, or optimize artifact naming.
- Do not redesign benchmark harness generation or `compare-perf` authority.
- Do not redesign the underlying `compare-perf` aggregate formulas in this change.
- Do not add static code checks that enforce which sections of the operator file may be edited.
- Do not allow `operator` mode to replace the Triton Ascend NPU computation with a pure PyTorch implementation.

## Alternatives Considered

### 1. Add `--optimize-target kernel|operator`

This keeps the behavior explicit and testable while reusing the current optimize workflow.

Pros:

- clear user-facing contract
- easy to document and test
- preserves backward compatibility through the default
- keeps one optimize command with one shared workflow

Cons:

- requires plumbing one additional field through CLI, request construction, prompts, and tests

### 2. Express operator-level optimization only through `--prompt`

This would avoid adding a formal CLI option and rely on appended user instructions.

Pros:

- smaller code change

Cons:

- not a real product contract
- hard to validate in tests
- easy for worker, resume, and supervisor prompts to drift out of sync
- poor discoverability in README help text

### 3. Add a separate operator-level optimize subcommand

This would create a second optimize command for the broader target.

Pros:

- very explicit at the command level

Cons:

- duplicates optimize surface area
- adds avoidable orchestration branching to the CLI
- conflicts with the repository preference for a thin CLI with explicit options

## Recommendation

Use alternative 1.

One explicit `--optimize-target` option is enough to express the semantic change without duplicating commands or hiding behavior inside free-form prompts.

## Design

### CLI Surface

Add `--optimize-target kernel|operator` anywhere optimize options are available today:

- `triton-agent optimize`
- `triton-agent optimize-batch`

Default the option to `kernel`.

That means:

- existing users keep today's kernel-focused behavior without changing their command lines
- users who want end-to-end operator optimization opt in with `--optimize-target operator`

### Optimize Modes

#### `kernel`

This mode preserves the current optimize meaning:

- optimize the Triton Ascend NPU kernel path itself
- wrapper code may remain as public API shape, but it is not the primary optimization target
- rounds that only improve wrapper behavior while avoiding meaningful kernel-path optimization should still fail the contract

#### `operator`

This mode broadens both the optimization target and the allowed change surface:

- optimize end-to-end operator latency
- allow coordinated changes across wrapper logic, data movement, scheduling, pre-processing, post-processing, and kernel implementation
- treat kernel work as a first-class optimization surface, not as an exceptional fallback
- still require the optimized operator to remain a Triton Ascend NPU-backed operator rather than a pure PyTorch replacement

The intent is not merely to change how results are evaluated. It is to explicitly permit whole-operator optimization work under one optimize contract.

### Compare-Perf Adaptation

The optimize target also changes how optimize should consume `compare-perf`.

#### `kernel`

In `kernel` mode, optimize should continue to prefer kernel-latency comparison, but it may use the existing `compare-perf --metric-source auto` behavior when kernel latency is unavailable for some cases.

That means:

- the optimize workflow should invoke `compare-perf` in kernel-oriented auto mode
- round records should store one resolved `effective_metric_source`
- the resolved value may be `kernel`, `total-op`, or `mixed`

This keeps kernel mode practical for imperfect perf artifacts while still making the final comparison basis explicit after the fact.

#### `operator`

In `operator` mode, optimize should ask for both views:

- kernel-facing comparison output for diagnosis
- total-op-facing comparison output for the official round conclusion

This is effectively an `all` analysis view: the agent should see both the kernel result and the total-op result for the same round so it can tell whether kernel improvements translated into end-to-end operator gains.

However, operator mode still needs one canonical basis for orchestration and status decisions. In this mode:

- total-op is the canonical comparison basis
- kernel comparison is diagnostic-only
- the canonical recorded `effective_metric_source` is still `total-op`

### Recorded Comparison Basis

Optimize session and round metadata should record exactly one comparison-basis field:

- `effective_metric_source`

This field should capture the actual basis used for the round conclusion:

- `kernel`
- `total-op`
- `mixed`

Do not add a second field such as `perf_metric_policy`. The optimize target already explains the intended mode, and `effective_metric_source` records the actual resolved result basis.

For `kernel` mode:

- `effective_metric_source` may be `kernel`, `total-op`, or `mixed`
- rounds that fall back away from pure kernel comparison may still pass and may still participate in best-round selection
- those rounds should surface a warning because the effective metric source no longer exactly matches the requested kernel-focused target

For `operator` mode:

- `effective_metric_source` should resolve to `total-op` for the official session conclusion
- kernel-side comparison remains available to the agent as analysis output, but it is not the canonical round basis

### Display Versus Decision

This change separates analysis display from orchestration truth.

In `kernel` mode:

- optimize output and round summaries should focus on the kernel-facing result
- if `compare-perf` falls back to `total-op` or mixes sources, that should be made explicit as a warning

In `operator` mode:

- optimize output and round summaries should show both kernel and total-op views
- the final conclusion should explicitly label total-op as the canonical basis

This lets agents reason from both signals in operator mode without making round ranking and resume semantics ambiguous.

### Aggregate Metric Display

`compare-perf` currently computes multiple aggregate metrics.

For this optimize-target change:

- keep computing `Avg improvement`, `Geomean speedup`, and `Total speedup`
- keep storing enough information for later use
- stop treating `Total speedup` as a required default display field in optimize-facing summaries for now

`Total speedup` remains available as internal data and may be surfaced again later when needed, but it should not drive the initial optimize-target UX by default.

### Prompt Contract Changes

Thread the selected optimize target through the optimize prompt builders and switch the contract language accordingly.

In `kernel` mode, keep the current lines about:

- continuing to optimize the Triton Ascend NPU kernel path itself
- rejecting rounds that bypass the Triton kernel path with pure PyTorch computation

In `operator` mode, replace the kernel-only requirement with wording that says:

- the optimization target is end-to-end operator latency
- the agent may optimize wrapper logic, data movement, scheduling, pre/post-processing, and kernel code together
- the agent must preserve a real Triton Ascend NPU computation path
- a pure PyTorch rewrite still does not count as a successful optimize round

This mode switch must apply consistently to:

- optimize worker prompts
- optimize unsupervised prompts
- optimize resume prompts
- optimize supervisor prompts

### Supervisor Audit Changes

Supervisor logic today is documented as rejecting rounds that keep only the public API shape while replacing the Triton kernel path with pure PyTorch computation.

That rejection rule should remain in both modes.

What changes is the positive audit standard:

- in `kernel` mode, the supervisor continues to audit against kernel-path-focused round intent
- in `operator` mode, the supervisor should accept validated rounds whose main improvement came from broader operator-level restructuring, as long as the Triton Ascend NPU computation path remains real

This keeps the anti-cheating rule stable while broadening what counts as a valid optimization direction.

When auditing performance conclusions, the supervisor should also respect the recorded `effective_metric_source`:

- in `kernel` mode, allow fallback-driven rounds to pass but surface a warning when the effective source is `total-op` or `mixed`
- in `operator` mode, treat total-op as the canonical basis for the round conclusion even when kernel diagnostics are also shown

### Workspace Guidance Changes

The temporary workspace guidance files used by optimize runs should also reflect the selected optimize target.

In particular:

- unsupervised guidance should state whether the session owns a kernel-focused optimize run or a whole-operator optimize run
- shared supervised guidance should mention the selected target so worker and supervisor runs do not reason from mismatched defaults

The guidance does not need to restate every prompt rule, but it should surface the selected target clearly enough that the launched agent sees the same contract in both the memory file and the launch prompt.

### Request And Runtime Plumbing

Add one explicit optimize-target field to optimize request construction so the selected mode can flow through:

- CLI parsing
- optimize run options
- optimize request creation
- prompt building
- resume prompt rebuilding
- temporary workspace guidance rendering

This field should live on the optimize path rather than being reconstructed indirectly from prompt text.

Add one explicit comparison-basis field to optimize session and round records:

- `effective_metric_source`

This field should be checked during resume and round validation so future optimize invocations do not silently reinterpret previous rounds under a different basis.

## Testing Strategy

Add or update tests for:

- CLI parsing defaults and accepted values for `--optimize-target`
- optimize option construction from parsed args
- prompt content in both `kernel` and `operator` modes
- supervisor prompt content in both modes
- resume prompt content in both modes
- unsupervised guidance file rendering in both modes when relevant assertions depend on the selected target
- optimize request construction so the selected target is preserved through orchestration
- optimize comparison-basis recording and validation
- warning behavior when kernel mode falls back to `total-op` or `mixed`
- dual-view operator-mode reporting that still records `total-op` as canonical

The tests should prove both:

- backward compatibility of the default `kernel` mode
- the broader contract wording in `operator` mode

## Expected Outcome

- users can explicitly choose whether optimize should target the kernel path or whole-operator latency
- default optimize behavior remains stable
- operator-level optimization becomes a supported, documented, and testable optimize mode
- worker, resume, supervisor, and workspace guidance all agree on the selected target
