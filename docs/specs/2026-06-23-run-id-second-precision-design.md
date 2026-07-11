# Run ID Second-Precision Design

## Goal

Make generated run IDs easier to read by default while preserving a simple collision fallback for repeated same-second allocations in one process.

## User-Visible Semantics

- Default generated run IDs should use second precision, for example `optimize-20260623-092959`.
- When the same process generates another run ID with the same prefix in the same second, append a short numeric suffix such as `optimize-20260623-092959-2`.
- Callers should keep using the shared helper; this change should apply consistently to optimize, generate, convert, log-check, and report run IDs.

## Design

- Keep run ID generation centralized in `src/helix/otel_trace.py`.
- Format the base timestamp with `%Y%m%d-%H%M%S`.
- Track per-process collisions by base run ID string and append `-2`, `-3`, and so on only when needed.

## Non-Goals

- Do not redesign workflow-state `run_id` semantics.
- Do not add random suffixes or process identifiers to the default first allocation.
- Do not change caller-facing `run_id` plumbing or environment variable names.

## Testing

- A first allocation in a second returns the base second-precision run ID.
- A second same-prefix allocation in the same second returns the same base plus `-2`.
- A no-prefix allocation follows the same rule.
