# Batch NPU Device Range Design

## Summary

Extend `HELIX_BATCH_NPU_DEVICES` parsing so users can mix explicit device IDs and closed numeric ranges such as `0,3-5,8-9` while preserving the existing batch affinity behavior after parsing.

## Goal

- Keep the environment variable name unchanged.
- Support comma-separated tokens that are either:
  - one explicit device ID such as `0`
  - one inclusive numeric range such as `3-5`
- Expand parsed ranges into the same ordered device tuple format already used by batch scheduling.

## Non-Goals

- Supporting stepped ranges such as `0-8:2`
- Supporting descending ranges such as `5-3`
- Supporting negative device IDs
- Changing batch leasing, env propagation, or concurrency validation behavior

## Desired Behavior

- `0,3-5,8-9` expands to `("0", "3", "4", "5", "8", "9")`
- Whitespace remains ignored around comma-separated tokens
- A token like `5-3` fails explicitly
- A token like `3-a` fails explicitly
- A token like `1-3-5` fails explicitly
- Duplicate devices are still rejected after expansion, so `0,0-2` fails because `0` appears twice

## Approach

Keep the change local to `src/helix/npu_affinity.py`:

1. Split the raw string on commas as today.
2. For each non-empty token:
   - if it contains no hyphen, keep it as one explicit device ID
   - if it contains one hyphen, require both sides to be non-negative integers and expand the inclusive range
   - otherwise fail
3. Run the existing duplicate check on the expanded device tuple.

This keeps every downstream caller unchanged because they already consume a tuple of string device IDs.

## Verification

- Add parser tests for mixed explicit IDs and ranges
- Add parser tests for invalid descending and malformed ranges
- Keep existing duplicate and empty-token tests green
- Run targeted unit tests for `tests/test_npu_affinity.py`
- Run full unittest discovery to confirm no batch-affinity regressions
