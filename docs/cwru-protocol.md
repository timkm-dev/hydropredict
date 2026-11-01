# CWRU fixed baseline protocol — issue #9

Owner requested CWRU implementation on 24 October 2026. This does not constitute
supervisor milestone approval. Public source: CWRU Bearing Data Center. Download
40 recordings: four normal baselines and 48 kHz drive-end faults, sizes 7/14/21
mils, loads 0/1/2/3 hp, inner race/ball/outer race at six o'clock. No raw files
are committed or redistributed; public availability is not an explicit license.

Use only the DE signal. Treat the normal baseline as 48 kHz (the common CWRU
baseline convention; its own download table omits the rate). Uniformly anti-alias
and resample every recording to 12 kHz, then use non-overlapping 4096-sample
windows. Save remaining sample counts. No filenames, load, RPM, fault size or
labels enter model features. Fit only on whole training recordings.

Primary: loads 0 and 1 train, 2 validation, 3 test. This evaluates different
recordings/loads of related seeded conditions, not independent bearings.
Additional challenge: withhold every 14-mil fault recording and normal load 3;
fit on 7/21-mil faults and other normal loads. All recordings from a withheld
fault-size/type family remain together. Physical bearing identity is not fully
specified, and healthy recordings may share the same bearing across loads.

Features: time-domain amplitude/shape statistics, normalized FFT band powers,
spectral centroid/entropy, and envelope-spectrum band powers. XGBoost fixed
200 trees, depth 3, learning rate .05, min-child-weight 3, lambda 2, seed 42,
two CPU threads. Per-recording and class balanced fitting weights. No tuning on
validation/test results and no training on final holdout for the served bundle.

Report recording-level mean-probability classification, per-class precision and
recall, macro-F1 and confusion matrix; window metrics are secondary and windows
are not independent trials. Save separate challenge bundle. Serve the primary
bundle as a laboratory benchmark even if challenge performance is weak, with
results visible. Display model support, not calibrated confidence or a physical
fault confirmation. This does not forecast failures or validate SHP performance.
