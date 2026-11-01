# CARE alert explanations: protocol

1 November 2026; related issue #21. Written before computing any explanation.
Question: when the autoencoder warns, do its top contributing sensors point to the
component that later failed? This tests whether alerts tell a technician where
to look, not whether the alert itself is correct.

## Explanation

- Use the frozen autoencoders and decision rule of
  [care-autoencoder-protocol.md](care-autoencoder-protocol.md) (6-hour smoothing).
- A sensor's contribution is its squared standardised reconstruction error,
  smoothed with the same causal 6-hour mean. Its share is that value divided by
  the sum over sensors at the same moment. The explanation is the top 3 sensors.
- Explained moments, per anomaly dataset:
  1. **Alarm**: the first CARE alarm (criticality reaches 72) up to the event
     end. Operationally realistic; only alarmed datasets have one.
  2. **Peak**: the reading inside the annotated event window with the highest
     smoothed total score. Uses the label to pick the moment, so it measures
     explanation quality only, never detection. Available for every anomaly dataset.

## Plausibility check (Farms A and B)

Fault component from the event description, by lower-case keyword, first match:
transformer ("transformer"), hydraulic ("hydraulic"), gearbox ("gearbox"),
generator ("generator bearing"), rotor bearing ("rotor bearing", "rotorbearing",
"main bearing").

Sensor component from its description, lower case: transformer ("transformer" or
"transformator"); hydraulic ("hydraulic"); gearbox ("gearbox"); generator
("generator" and "temperature"); rotor bearing ("rotor bearing").

A hit means at least one top-3 sensor matches the fault component. Chance is the
probability that 3 sensors drawn at random from that dataset's model inputs
include a match. Compare observed hits with the expected chance count, with an
exact one-sided Poisson-binomial p-value. Datasets whose fault component has no
matching input sensor are reported but excluded from the check.

Farm C fault descriptions are free text in mixed languages, so Farm C gets
worked examples only, after its confirmation run. No settings change after results.
