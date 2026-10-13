# SHP readiness audit

13 October 2026. Decision: **conditional GO for an exploratory numerical Isolation Forest
baseline; not yet a validated operational warning system**.

## Source findings

The source publication describes four vibration measurements, HLU lubrication inflow and
generator apparent power. It does not establish the code's temperature/active-power labels
or the exact V1..V6 mapping and engineering units. Physical metadata is therefore unset
until a source data dictionary is verified. Existing domain features remain experimental;
they are excluded from this baseline and must not support physical diagnoses.

The authors constructed the published test set from 12-hour pre-fault windows and treated
the remaining source training observations as normal. Thus the local training subset has
documented normal-reference provenance, but inherits retrospective source selection.
No additional fault-based filtering is applied to training or calibration. The chronological
cutoffs remain unchanged. Naive timestamps retain their common UTC computational convention;
the real plant timezone remains unresolved, so absolute timezone claims are inappropriate.

Sources: [de Santis and Costa 2020, Section 2.1](https://doi.org/10.3390/su12166421),
[accessible copy of the original paper](https://www.wisdomlib.org/uploads/journals/mdpi-sust/2020-volume-12-issue-16--2071-1050-12-16-6421-.pdf),
[authors' data repository](https://github.com/rodrigosantis1/shp_anomaly).

## Computed evidence

All source and partition hashes matched the verified preparation manifest. Training,
validation and testing are ordered chronologically and contain 30,807, 605 and 733 original
observations. Fault timestamps do not enter model features, fitting or threshold calibration.

For each event, inspect the six hours strictly preceding the fault. Eligibility requires
at least 80% of 72 unique five-minute slots and a latest observation within ten minutes
of the event. These declared research assumptions are not an external standard. Coverage
uses original observations, so filled rows cannot inflate it. Duplicate slots count once.

| Partition | Faults in declared calendar interval | Any pre-fault data | Eligible under policy |
| --- | ---: | ---: | ---: |
| Validation | 9 | 8 | 6 |
| Final test | 11 | 11 | 5 |

Report the full per-event audit, excluded events and sensitivity to this policy alongside
recall. Five eligible test events are too few for a strong generalisation claim. An alert
within a window does not establish fault diagnosis or future plant performance.

Pre-fault excerpts cannot establish a general false-alert frequency. Later observations
from the publisher's normal-reference source may be scored separately as a retrospective
nominal reference; report their exceedance rate with its selection bias. Do not describe
unmatched warnings in fault-adjacent excerpts as confirmed false alarms.

## Reproduction

```powershell
python scripts/audit_shp.py --preparation-run data/processed/runs/2026-10-13-verified --output data/processed/runs/2026-10-13-verified/readiness_audit.json
```

The audit is retrospective evaluation code. The next baseline uses only the six raw
channels and a frozen training-derived scaler, avoiding the unresolved physical features
and offline gap-insertion policy. Calibration uses a later, disjoint normal-reference
subset preceding validation. No threshold or model tuning uses held-out faults.
