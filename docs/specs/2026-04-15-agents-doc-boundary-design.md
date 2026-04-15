# AGENTS Document Boundary Design

## Context

`AGENTS.md` had accumulated four different kinds of content:

- stable project rules
- design preferences
- feature-specific implementation constraints
- command-level operational details

That mix made `Core Principles` longer than it needed to be and made the document harder to use as a durable policy reference.

## Decision

Restructure `AGENTS.md` into a shorter policy document with a clearer boundary:

- keep stable project identity and decision-making rules in `AGENTS.md`
- compress repeated design preferences into a few high-level principles
- keep feature semantics, command examples, and verification commands in `README.md` or focused docs
- keep optimize-specific policy at the level of durable workflow expectations, not implementation layout details

## Resulting Structure

`AGENTS.md` should focus on:

1. project overview
2. core principles
3. workspace and backend rules
4. documentation and verification expectations
5. scope guardrails

## Non-Goals

- changing CLI behavior
- changing optimize semantics already documented elsewhere
- removing durable project constraints that still guide agent decisions
