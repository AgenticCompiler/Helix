# Round Operator Python Discovery

## Summary

Round operator discovery must only treat Python source files as fallback operator
candidates. Non-Python artifacts in an `opt-round-*` directory are not operator
inputs and must not be statted during this fallback scan.

## User-Visible Semantics

- The expected `opt_<operator>.py` name and the legacy source-operator name remain
  the first two lookup choices.
- If neither name is available, the fallback chooses from regular `.py` files only.
- JSON, Markdown, performance, profile, and other non-Python artifacts cannot be
  selected as the round operator or cause an access failure while candidates are
  being discovered.

## Scope

- Update only the fallback candidate filter in the optimize-state round checker.
- Add a regression test that simulates an unreadable JSON artifact and verifies the
  Python fallback still resolves.
- Do not change round-state contracts, artifact layout, or the named-operator lookup
  behavior.
