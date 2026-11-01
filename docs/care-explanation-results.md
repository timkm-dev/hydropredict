# CARE alert explanations: results (Farms A and B)

1 November 2026; related issue #21. Method frozen beforehand in
[care-explanation-protocol.md](care-explanation-protocol.md).

**Result: explanations point to the failed component more often than chance,
but not reliably enough to act on alone.** At each fault's strongest deviation,
the top 3 sensors included the failed component in 7 of 18 faults, against 3.3
expected by chance (p = 0.023). At the actual alarm moment the rate was 3 of 8,
against 1.6 by chance (p = 0.18): in the right direction, too few alarms to be sure.

## What an explanation looks like

Each alert lists the sensors that the normal-behaviour model reconstructs worst,
as shares of the total deviation over the last 6 hours. Examples:

| Fault (annotated) | Moment | Top 3 sensors (share of deviation) | Points to the fault? |
|---|---|---|---|
| A 68: Transformer failure | peak, 103 h before end | HV transformer phase L1 (29%), hydraulic oil (9%), VCS choke coils (6%) | Yes |
| A 40: Generator bearing failure | peak, 131 h before end | **Generator bearing 1 (63%)**, hydraulic oil (11%), gearbox oil (7%) | Yes |
| B 77: Rotor bearing damage | alarm, 689 h before end | Tower natural frequency (28%), generator bearing 1 (11%), **rotor bearing 2 (7%)** | Yes, 29 days early |
| B 27: Rotor bearing damage | alarm, 843 h before end | Generator bearing 1 (26%), tower frequency (18%), **rotor bearing 1 (13%)** | Yes, 35 days early |
| A 0: Generator bearing failure | alarm, 372 h before end | HV transformer L1 (40%), L2 (17%), L3 (9%) | No |

Full table: `docs/results/care-autoencoder-v1/explanations-A-B.csv`.

## Results

| Explained moment | Faults explained | Top 3 includes failed component | Expected by chance | One-sided p |
|---|---:|---:|---:|---:|
| First alarm | 8 | 3 | 1.55 | 0.18 |
| Strongest deviation in fault window | 18 | 7 | 3.26 | 0.023 |

By farm, Farm B is stronger: all 3 rotor-bearing faults name a rotor bearing
temperature among the top 3, alongside tower vibration or tower frequency, which
is physically plausible for main-bearing damage. Farm A's hydraulic faults are
never matched: one hydraulic oil sensor among 42 gives little chance, and those
faults may not show in temperature at all.

## Limitations seen in the results

- **Recurring sensors.** Farm A's HV transformer temperatures lead the explanation
  for several non-transformer faults (A 0, A 73). They deviate easily, so they can
  crowd out the real cause. Explanations show where behaviour differs from normal,
  not proven root cause.
- **Operating conditions.** Wind speed and grid quantities sometimes rank highly
  (A 45, B 34). These describe conditions outside the training range, not
  component health. Showing them apart from health sensors would help a technician,
  but changing the rule now would be tuning on these results; any such variant must
  be checked on Farm C or new data.
- **Small samples.** 18 faults, several repeated turbines, keyword matching of
  free-text fault labels. Treat the p-values as indicative.

## How this fits the system

The model warns, then names the sensors to inspect first, and the operator
decides. This is decision support with an evidence trail, not automatic diagnosis.
Farm C gets worked examples after its confirmation run.
