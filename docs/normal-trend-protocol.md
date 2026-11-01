# Normal-trained trend experiment

Declared before this run on 22 October 2026. The user authorized one additional
trend experiment after the supervised hybrid study. Earlier stopping rules remain
historical; this is a new, explicitly scoped development study. GitHub is currently
unavailable, so no issue has been read or updated for this work.

Use only the checksum-verified original Colab input package: earlier normal
reference for fitting/calibration, February–April for development. No final test.
Keep the frozen LSTM and dashboard unchanged. No threshold or feature searches.

Features: six elapsed-hour regression slopes and sample standard deviations for
all six channels. No interpolation; reset after gaps over 450 seconds; require
six elapsed hours since segment start and 58 distinct five-minute slots. Slopes
use actual elapsed hours. Prior observed history is allowed at prediction time.

Fit means and standard deviations of the 12 trend features on features from the
first 80% of earlier normal readings. Build calibration features independently
from the last 20%, discarding their initial warm-up. Score by maximum absolute
standardized feature deviation, with fitting scales floored at 1e-8. At least
100 fitting and 100 calibration feature rows are required, otherwise stop.

Calibrate trend threshold at the 99th percentile of earlier calibration scores.
For combined scoring, take the maximum of trend/threshold and frozen LSTM
score/threshold; calibrate its threshold at the 99th percentile on matching
earlier calibration endpoints. Use strict exceedance. Apply three-reading
persistence to trend and combined scores, restarting across gaps over 450
seconds. Three readings span about ten minutes at five-minute cadence.
The LSTM comparator retains its original point-alert decisions.

Compare all three on identical endpoints. Also report LSTM native availability,
all nine faults, original and matched coverage, alert episodes, nominal point
fractions and observed exposure. Use the repository six-hour/80%/latest-within-
ten-minutes eligibility policy. Overlapping precursor windows form one group.
Require at least one more eligible group warned than LSTM, no loss of LSTM
eligible group warnings, and at most 2% matched selected nominal point alerts
to call a result promising. This does not establish operational false-alarm rate.

Save input hashes, model statistics, thresholds, predictions and result checksums.
Verify prefix invariance, gap behavior and saved-statistic replay. Stop after
this fixed experiment regardless of outcome. February 3 was inspected before
design; any success remains hypothesis development requiring later independent
evaluation.
