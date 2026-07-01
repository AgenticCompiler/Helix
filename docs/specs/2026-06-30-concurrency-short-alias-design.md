# Concurrency Short Alias Design

## Goal

Add a short `-c` alias anywhere the CLI already exposes `--concurrency`, so
batch-oriented commands can use a shorter, consistent flag form without
changing existing semantics.

## User-Visible Semantics

- Commands that already accept `--concurrency` also accept `-c`.
- `-c` and `--concurrency` are exact aliases for the same parsed value.
- Existing validation rules stay unchanged:
  - commands that accept `max` continue accepting it with `-c max`
  - commands that require a positive integer still reject `-c max`
- Commands without a `--concurrency` option do not gain a new `-c` flag.

## Scope

In scope:

- the shared parser branch that adds `--concurrency`
- parser regression coverage for commands that use that shared branch
- README wording updates for user-facing batch command docs

Out of scope:

- changing concurrency defaults
- changing validation rules
- adding short aliases for unrelated numeric flags such as `--report-workers`

## Design

The CLI already centralizes `--concurrency` option creation behind
`spec.concurrency_default`. Extend that one `argparse` call to register both
`-c` and `--concurrency`. Because parsing, validation, and handler plumbing
already key off `args.concurrency`, no downstream code changes are needed.

Add parser coverage proving representative commands accept `-c`, including both
the `max`-accepting and integer-only variants, then update README batch option
lists to advertise the new short alias.
