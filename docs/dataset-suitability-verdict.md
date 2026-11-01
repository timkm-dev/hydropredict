# Dataset suitability verdict

24 October 2026. **Keep this dataset. Rebuild the supervised experiment using
the earlier fault excerpts before considering replacement data.** The full files
contain useful earlier fault history that the normal-only Colab package omitted.
This corrects the earlier implication that insufficient pre-February fault history
in that package necessarily reflected insufficient history in the full dataset.

This investigation does not establish reliable predictive performance. It
establishes a feasible, better-grounded development experiment and identifies
which claims the available data cannot support.

## What was verified

All three local raw-file hashes match `data/dataset_manifest.csv`. Their timestamp
union contains 52,754 unique observations: 47,857 publisher-normal readings and
4,897 pre-fault readings. There are 59 recorded fault timestamps. Coverage inventory
reads only timestamps across the full period; it neither scores later data nor
loads later measurement columns. Signal comparison uses only the existing
earlier-normal and February–April development package.

GitHub access was subsequently restored through Git Credential Manager and the
REST API. Issues #19 and #20 track the corrected package and fixed supervised
comparison. The authors' public repository and original paper were read.

## What the labels mean—and what remains unknown

The authors describe fault events identified from status reports and a maintenance
management system. Maintenance, planned stops and operator interventions were
filtered out of the telemetry. Twelve-hour pre-fault excerpts became their test
file; the remaining observations formed their normal-reference training file.
The six channels cover vibration, lubrication inflow and apparent power.

The available fault CSV contains timestamps only. It does not provide failure
mode, repair, cause, event duration or a definition distinguishing failure onset
from trip/record-entry time. The later [author-code review](shp-source-code-review.md)
establishes V6 as apparent power, V5 as lubrication inflow and V1-V4 as vibration.
Individual vibration orientations, engineering units and source timezone remain
unverified. Those omissions still limit physical diagnoses.

The paper's nearest-event absolute temporal distances and detection-count
comparison differ from our chronological six-hour warning rule. Its published
results therefore do not establish our target performance. Our six-hour horizon
is a project assumption requiring engineering justification, not a source label.

Source: [original paper, Sections 2.1 and 2.5](https://www.wisdomlib.org/uploads/journals/mdpi-sust/2020-volume-12-issue-16--2071-1050-12-16-6421-.pdf),
[authors' repository](https://github.com/rodrigosantis1/shp_anomaly).

## Full history inventory

| Period | Fault timestamps | Adequate raw six-hour coverage | Adequate seven-reading sequence coverage | Adequate six-hour continuous-context coverage |
| --- | ---: | ---: | ---: | ---: |
| Before February 2019 | 39 | 28 | 21 | 14 |
| February–April development | 9 | 6 | 3 | 2 |
| May onward | 11 | 5 | 4 | 3 |

Eligibility uses 80% of expected five-minute slots and a reading within ten
minutes. Sequence availability requires seven readings with 150–450-second
steps. These are full-union availability counts, not trained-model performance;
new chronological boundaries and feature requirements can reduce them.

The 21 sequence-eligible earlier timestamps span **18 non-overlapping precursor
groups**. These groups reduce double counting but do not establish independent
physical failures. Earlier six-hour-context events span 14 such groups. Thus the
earlier history is materially larger than the hybrid's development-only training
pool, although still small for generalization claims.

The original LSTM deliberately used earlier normal observations alone. Earlier
pre-fault source rows were excluded from that training package. That design was
appropriate for normal-only learning; reusing the same package for supervised
training inadvertently excluded potentially useful labelled examples.

## Do the signals contain a distinguishable precursor?

A descriptive comparison selected 402 non-overlapping earlier normal six-hour
windows. For each eligible development event, its first-hour six-channel mean
was compared with each control's first-hour mean in fitting-standard-deviation
units. The nearest 20 were retained, without selecting on the later slope. The
six-hour slopes then used actual elapsed time. No detector or threshold was fitted.

| Event | Observation against the selected normal controls | Interpretation |
| --- | --- | --- |
| February 3 | V1–V4 changes are +1.66, +1.89, +1.83 and +2.39 fitting SD; none of the 20 controls has an equally large absolute slope for those channels | A distinctive local trend exists in this comparison; the normal-trained global detector did not exploit it |
| April 2, 09:06 | 30–95% of controls have equally or more extreme absolute channel slopes | That fault does not share the same strong six-hour trend pattern |
| February 2 | V4's decline exceeds all 20 matched absolute slopes; other channels overlap controls | Another descriptive pattern, not evidence of a common failure mechanism |

Zero of 20 is not a zero false-alarm probability or statistical significance.
Events and features were previously inspected. Controls are publisher selected
and earlier in time; matching numerical starting levels does not verify equal
load, operating state or maintenance condition. Channels are correlated. The
comparison cannot prove that the signal changes were caused by degradation.
Windows require coverage but need not be uninterrupted; they are descriptive
summaries, not the strict continuous-history detector used previously.

The evidence supports neither "no predictive information exists" nor "trends
solve prediction." It supports testing labelled, chronological examples from
the earlier history with explicit coverage and operating-state limitations.

## Concrete next experiment

1. Build an exact timestamp union of earlier normal and pre-fault source files,
   preserving source provenance for auditing only. Check duplicate conflicts and
   never interpolate unavailable measurements. Include all earlier fault labels.
2. Use August–December for supervised fitting, January for calibration, and
   February–April for development evaluation. Inventory indicates 31 earlier
   fault timestamps with 18 sequence-eligible timestamps across 15 groups before
   January; January has eight timestamps with three sequence-eligible groups.
   Recheck eligibility after enforcing split boundaries. Purge target windows
   crossing boundaries and keep overlapping event windows together.
3. Predeclare one compact feature set with short history, a single class-weighted
   classifier configuration and the existing 2% selected nominal alert budget.
   Keep the six-hour warning target for comparability; do not choose a new horizon
   based on which makes February 3 succeed. Fit all preprocessing before calibration.
   A rolling evaluation must also refit the LSTM if its training would otherwise
   include observations later than that fold.
4. Compare against frozen LSTM scores on identical development endpoints, reporting
   native availability too. Keep all fault outcomes, nominal episodes and lead
   times. Distinguish fault-proximity labels from confirmed health labels.
5. Stop after the predeclared run. Keep a candidate only if it adds covered event
   groups without losing baseline detections and stays within the alert budget.
   Failure would justify retaining the monitoring prototype and seeking more
   informative event/operating context—not another search over the same faults.

May-onward data was inventoried by timestamps only in this investigation. Earlier
repository work already reported test outcomes; it cannot simply be relabelled
an untouched final test. Independent confirmation needs a documented evaluation
plan and genuinely uninspected data. This limitation does not prevent the proposed
development experiment or completion of an honest monitoring proof of concept.

## Reproduction and checks

```powershell
.venv/Scripts/python.exe scripts/assess_dataset_suitability.py --output models/dataset-suitability-new-run
```

Canonical run: `models/dataset-suitability-2026-10-24-verified`. CSV evidence is
also saved under `docs/results/dataset-suitability-2026-10-24/`. The test checks
elapsed-time slope calculation, first-hour matching, exclusion of the fault
instant, and rejection of insufficient coverage/latest observations. Lint and
format checks pass. Source hashes and output checksums accompany the run.
