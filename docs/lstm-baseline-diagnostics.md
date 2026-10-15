# First LSTM baseline diagnostics

15 October 2026. The completed Colab LSTM export remains unchanged. Local CPU replay
reproduced all exported alert decisions and scores within floating-point tolerance.
Bundle checksums and packaged input hashes were verified. No final test was evaluated,
no weights were retrained, and no candidate was promoted.

## Fair comparator results

The original raw Isolation Forest and the later 30-minute temporal Isolation Forest are
different comparators. The earlier statement that both models detected February 2 refers
to the 30-minute comparator: the original raw baseline did not detect that event.

All models were scored on the same reconstructed February-April development timeline.
On the 5,887 selected normal-reference timestamps scored by all three:

| Model | Point alerts | Fraction | Alert episodes | Adequately covered detections |
| --- | ---: | ---: | ---: | ---: |
| Original raw Isolation Forest | 97 | 1.65% | 70 | 0 / 3 |
| 30-minute Isolation Forest | 60 | 1.02% | 32 | 1 / 3 |
| First completed LSTM | 79 | 1.34% | 25 | 1 / 3 |

The LSTM flags more points but fewer episodes than the 30-minute model. Both detect the
same February 2 fault. These fractions describe publisher-selected normal-reference
readings, not proven real-world false-alarm rates. Episodes are consecutive alert readings
with gaps no greater than 450 seconds; matched-row filtering can change episode counts.
Among the original six eligible fault records, the LSTM detects one. Only three retain
adequate scored coverage. The other denominator is reported separately, never substituted
to imply higher general recall. April 2 fault records have overlapping warning windows.

## Missed faults and warning interpretation

The LSTM threshold is 0.30444768. Maximum six-hour scores are 0.18596 before February 3,
0.14069 before March 15 at 16:06, and 0.20489 before the April 2 records. These are about
61%, 46%, and 67% of threshold, respectively. March 15 and the later April 2 records lose
adequate scored coverage, so unavailable sequences remain a separate limitation.

Before February 3, V4 error briefly reaches approximately the 99.54th percentile of the
development normal-reference distribution; three of 71 scored readings exceed its 99th
percentile. The other channels do not exceed their corresponding 99th percentiles.
This motivated one experiment on channel dilution. The development percentiles are
diagnostic descriptions only; candidate calibration uses earlier normal data exclusively.
March 15 and April 2 have no scored channel errors above those development 99th percentiles.
This does not establish that their raw readings contain no useful predictive information.

The February 2 LSTM detection is a single alert at February 1, 18:12:45 UTC, 354.24 minutes
before the fault. It is the first eligible sequence in the available timeline, beginning
at 17:42:45. V6 error is 1.25835, then drops to 0.09323 at the next endpoint. V4 is also
elevated. The plotted raw readings show changing operating values around this period.
The 30-minute model alerts approximately 19.24 minutes before the same recorded fault.
The isolated LSTM crossing qualifies under the fixed event rule, but its position at the
start of available context and lack of persistent alerts prevent treating the lead time
as convincing evidence of a sustained precursor. A startup artifact or causal fault
relationship has not been established. Reconstruction errors are not causal attributions.

## One completed experiment

Protocol was written before scoring the candidate. With all network weights frozen,
divide each channel error by that channel's 99th-percentile error on the original normal
calibration sequences. Take the maximum across all six channels, then calibrate a joint
99th-percentile threshold on those same normal sequences. Joint calibration prevents
mistaking six separate 1% channel tests for a single 1% alert rule. No channels were chosen
or weighted from fault outcomes, and no threshold grid was searched.

Candidate threshold: 1.846165. It still detects only the February 2 fault among the three
adequately scored events. February 3 and April 2 maximum scores remain below its threshold.
Matched nominal alerts increase to 101/5,887 (1.72%), across 28 episodes. The candidate
fails the declared requirement of an additional adequately scored detection within a 2%
normal-reference point-alert budget. Retain the candidate as negative development evidence;
do not replace either baseline or lower its threshold after seeing this result.

## Saved evidence and reproduction

Diagnostics: `models/lstm-diagnostics-2026-10-15-v1` contains nine event plots, complete event
and channel summaries, matched/native model metrics, calibration scores, replayed scores,
a checksum-verified bundle copy and output checksums. Candidate outputs and protocol:
`models/lstm-channel-score-2026-10-15-v1`. Model folders are local ignored artifacts; keep
them with the unchanged original ZIP when archiving this milestone.

```powershell
.venv/Scripts/python.exe scripts/diagnose_lstm.py --export 'C:/Users/tmuny/Downloads/hydropredict-lstm-initial-v1-20261015T041243Z-1-001.zip' --inputs artifacts/colab/hydropredict-lstm-input-2026-10-14.zip --output models/lstm-diagnostics-new-run
.venv/Scripts/python.exe scripts/experiment_lstm_channel_score.py --diagnostics models/lstm-diagnostics-new-run --output models/lstm-channel-score-new-run
```

The experiment requires the same original packaged input at the recorded repository path
and verifies its hash. Repeated use of this development interval creates selection bias;
these results support local diagnosis, not an independent performance claim.

Next, investigate operating regimes and whether the missed, adequately observed faults
show a pre-fault raw-channel change distinguishable from normal operation. Resolve sensor
semantics and available maintenance context before choosing another model experiment.
Additional epochs alone do not address the demonstrated detection limitation. Establish
an independent evaluation plan before making predictive-usefulness claims.
