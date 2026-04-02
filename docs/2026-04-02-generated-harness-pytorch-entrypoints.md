# Generated Harness PyTorch Entrypoints

## Summary

- Extend `gen-test` and `gen-bench` so generated harnesses no longer assume every operator file exposes a Triton wrapper function as the only public API shape.
- Support three entrypoint kinds for generated harness metadata and runtime loading:
  - `triton-wrapper`
  - `torch-function`
  - `torch-module`
- Keep `--operator-file` as the only runtime operator path input while making the generated harness smart enough to load and invoke the resolved entrypoint correctly.

## User-Visible Behavior

- `gen-test` and `gen-bench` should work for both existing Triton-wrapper operators and mixed Triton + PyTorch operator files.
- When the operator file exposes a plain PyTorch function that internally calls Triton kernels, the generated harness should treat that PyTorch function as the API under test or benchmark.
- When the operator file exposes a `torch.nn.Module` class that represents the operator or model entrypoint, the generated harness should record that class as the API and instantiate it at runtime before invoking it.
- Generated harness metadata should declare not only the resolved API name but also how that API should be invoked.
- Existing wrapper-based operator files should continue to generate the same class of harnesses with no behavior change beyond the richer metadata header.

## Metadata Contract

Generated test and benchmark harnesses should include:

- `# test-mode: <mode>` or `# bench-mode: <mode>`
- `# api-name: <resolved-entrypoint-name>`
- `# api-kind: <triton-wrapper|torch-function|torch-module>`
- `# kernel: <resolved-primary-triton-kernel>`

`api-name` remains the stable identifier for the entrypoint symbol in the runtime operator file.

`api-kind` tells the generated harness how to turn that symbol into an invokable callable:

- `triton-wrapper`: load the named symbol directly and call it as a function
- `torch-function`: load the named symbol directly and call it as a function
- `torch-module`: load the named symbol as a class, instantiate it, then call the resulting module instance

## Runtime Loading Contract

- Generated harnesses should continue loading `--operator-file` with `importlib`.
- Runtime should not re-infer entrypoint type from the file contents; the embedded metadata remains the source of truth.
- Runtime loading behavior should be:
  - `triton-wrapper` -> `getattr(module, api_name)`
  - `torch-function` -> `getattr(module, api_name)`
  - `torch-module` -> `module_cls = getattr(module, api_name)` then `module_cls()`
- For `torch-module`, the initial scope should support only no-argument construction.
- If a `torch-module` entrypoint cannot be instantiated without arguments, the generated harness should fail with an explicit actionable error instead of guessing constructor inputs.

## Skill Contract Updates

### `skills/test-gen/SKILL.md`

- Replace the wrapper-only assumption with an entrypoint-resolution contract.
- Require the skill to resolve both:
  - the public operator entrypoint
  - the entrypoint kind
- Update workflow and failure handling to allow PyTorch functions and module classes as valid targets.
- Keep raw Triton kernel functions disallowed as direct test targets.

### `skills/bench-gen/SKILL.md`

- Mirror the same entrypoint-resolution contract as `test-gen`.
- Allow benchmark targets to be wrapper functions, PyTorch functions, or no-argument `torch.nn.Module` classes.
- Keep benchmark generation focused on the public operator entrypoint, not raw kernels.

## Spec Document Updates

Update these normative spec files so they no longer describe only wrapper APIs:

- `skills/test-gen/references/test-standalone-spec.md`
- `skills/test-gen/references/test-differential-spec.md`
- `skills/bench-gen/references/bench-standalone-spec.md`
- `skills/bench-gen/references/bench-msprof-spec.md`

Required changes:

- Replace “wrapper API” wording with “resolved operator entrypoint”.
- Add `# api-kind:` to the required metadata header.
- Define the runtime loading behavior for each supported `api-kind`.
- Update the sample header blocks and sample loader code.
- Keep `# kernel:` in place because msprof and related tooling still need the Triton kernel identity.

## Selection Rules For Generators

The generator should resolve entrypoints using this precedence:

1. A clear public Triton wrapper function when present and obviously intended as the operator API.
2. Otherwise, a clear public PyTorch function/operator that represents the user-facing callable.
3. Otherwise, a clear `torch.nn.Module` class that represents the operator or model entrypoint and supports no-argument construction.

Stop and explain the ambiguity instead of guessing when:

- multiple plausible public entrypoints exist without a clear best choice
- the only plausible module entrypoint requires constructor arguments that cannot be inferred safely
- the file contains only raw Triton kernels and no higher-level public API

## Benchmark-Specific Notes

- `standalone` benchmarks can use any supported `api-kind` as long as the generated harness can obtain a callable.
- `msprof` benchmarks still require a stable Triton kernel name for profiler invocation.
- If the file exposes a valid PyTorch entrypoint but the primary Triton kernel cannot be resolved safely for msprof mode, benchmark generation should fail explicitly for msprof rather than silently emitting a broken harness.

## Documentation Updates

- Update the generated harness metadata design doc to describe `api-kind`.
- Update `README.md` examples and narrative so “API” means the resolved public entrypoint, not only a wrapper function.
- Update `AGENTS.md` to state that generation skills must support Triton-wrapper and Triton-plus-PyTorch entrypoint patterns.

## Tests

- Extend generation contract tests to require `# api-kind:` in both skills and all normative spec files.
- Replace assertions that hard-code `resolved_wrapper_api` with neutral `resolved_entrypoint` wording.
- Add tests for the documented runtime loading contract for `triton-wrapper`, `torch-function`, and `torch-module`.
- Add tests that the specs explicitly reject guessing constructor arguments for `torch-module`.

## Scope

- This change is about generation-side contracts and generated harness behavior.
- Do not redesign `run-test` or `run-bench` CLI flags in this change.
- Do not support `torch-module` constructors with required parameters in this first version.
- Do not treat raw `@triton.jit` kernels as direct public APIs for generated tests or benchmarks.
