# Deterministic Generation Inputs Design

## Goal
Make `gen-test` and `gen-bench` generation reproducible by requiring seeded random input construction when randomness is used.

## Approach
Update the skill docs, benchmark/test specs, and CLI prompt text to require explicit seed control inside case construction. The contract should allow random tensor generation, but repeated runs of the same generated harness must produce identical inputs for the same case parameters.

## Verification
Add contract tests that lock the new wording in the skills, specs, and prompt builder so the requirement cannot regress silently.
