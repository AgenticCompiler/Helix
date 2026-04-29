# Symptom Index

Use this index after structured profile or IR evidence already exists.

Start here to identify the dominant symptom, then read only the one or two matching symptom cards before returning to detailed pattern references.

## Available Symptom Cards

- [high-scalar-overhead](high-scalar-overhead.md): Many tiny programs, scalar control work, or low useful work per launch dominate the round.
- [high-transfer-pressure](high-transfer-pressure.md): Data movement, staging, or memory traffic dominates more than useful compute.
- [poor-locality](poor-locality.md): The kernel repeatedly touches data in a reuse-unfriendly order or keeps colliding on the same cache regions.
- [weak-pipeline-overlap](weak-pipeline-overlap.md): Memory movement and compute appear insufficiently overlapped, leaving avoidable wait.

## Routing Notes

- Symptom cards are routing aids, not diagnosis truth.
- Use them only after profile summaries or IR signal summaries exist.
- A symptom card should narrow detailed pattern reading, not replace operator-specific reasoning.
