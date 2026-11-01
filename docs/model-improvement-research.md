# Model improvement investigation

Researched 24 October 2026 against primary papers, author implementations and
official documentation. Scope: find materially different ways to improve
six-hour fault warning on the existing SHP data.

**Decision: investigate weakly supervised precursor-pattern learning first,
with MiniROCKET as the sequence-classification comparator.** These are research
candidates, not demonstrated improvements. No new model was trained during this
literature investigation. The prior conclusion that richer metadata was the
only substantive next step was too restrictive: several relevant model families
remain untested.

## What the failures establish

The corrected supervised model warned before 0/3 eligible February-April groups;
the LSTM warned before 1/3. Four absolute/relative summary-feature classifiers
also failed the fixed November/December transfer criteria. See
[the fixed comparison](supervised-precursor-results.md) and
[earlier backtests](precursor-transfer-results.md).

This rules out those configurations as improvements. It does not rule out
supervised sequence classification, weak-label learning, learned representations
or conditional sensor models. Our LSTM learns normal reconstruction; it does
not directly learn which evolving patterns distinguish impending faults.
The other classifiers compress history into short summaries.

Two unresolved hypotheses are especially worth testing:

1. A useful precursor might occupy only a small part of the six-hour interval.
   The six-hour proximity labels are correct for the declared target, but they
   do not establish that every positive endpoint contains a recognizable local
   precursor. A model trained equally on all those endpoints may learn poorly.
2. Faults may have several recurring precursor shapes rather than one common
   increase, threshold or operating level. Aggregated means/slopes can hide
   ordering and channel interactions. These are hypotheses, not diagnosed causes.

## Ranked shortlist

### 1. Weak-label shape dictionaries and contrast profiles

SDTS learns dictionaries of time-series patterns under imprecise annotations,
class imbalance and varying pattern lengths. The contrast profile identifies
subsequences similar to one collection but dissimilar to another.
[SDTS paper, VLDB 2017](https://www.vldb.org/pvldb/vol10/p1802-yeh.pdf);
[contrast-profile paper, 2022](https://link.springer.com/article/10.1007/s10618-022-00824-5).

**Project adaptation:** look for short shapes repeated in different earlier
precursor groups and absent from selected-normal controls. At inference, compare
only the trailing observed sequence against the frozen dictionary. This targets
recurring precursors rather than generic unfamiliarity. The adaptation would be
our model, not a claim to reproduce the published algorithms unchanged.

**Why first:** it accommodates uncertain precursor location, can retain several
different patterns, and permits a concrete explanation: which historical pattern
matched, in which channel, and before which distinct training events.

**Main risks:** ordinary operating cycles may look distinctive in a small sample;
similar overlapping windows from the same event can create false support. Exclude
the entire source event group when measuring recurrence. A fault without an
observable recurring precursor cannot be rescued by this approach.

**Implementation:** STUMPY provides matrix-profile, motif and matching primitives;
it is not a turnkey implementation of our proposed event-aware learner.
[Official STUMPY documentation](https://stumpy.readthedocs.io/en/latest/).
An exact NumPy/SciPy prototype is also practical for a restricted candidate bank.

### 2. MiniROCKET, followed by MultiROCKET only if warranted

MiniROCKET transforms raw sequences with convolutional kernels and feeds the
result to a linear classifier. MultiROCKET adds first differences and additional
pooling operators. This preserves more sequence structure than the 36 rolling
summary features we tested.
[MiniROCKET author implementation](https://github.com/angus924/minirocket);
[MultiROCKET paper](https://arxiv.org/abs/2102.00457).

**Project adaptation:** six-channel trailing sequences, transformed using fitting
data only, followed by a strongly regularized classifier. Weight precursor groups
so a long recording does not dominate through hundreds of correlated examples.
Do not use random cross-validation on overlapping windows.

The maintained aeon transformer supports multivariate, equal-length input but
not missing values; it uses length-nine kernels. Our existing seven-reading LSTM
windows therefore cannot be reused unchanged. A 13-reading window is a sensible
initial candidate: nominally one hour, with actual elapsed span recorded and
cadence enforced. It may reduce coverage and must be compared on shared endpoints.
[Current aeon API](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.transformations.collection.convolution_based.MiniRocket.html).

Use a fixed classifier configuration or chronological inner selection instead
of blindly using the default RidgeClassifierCV pipeline. Benchmark success on
generic time-series classification is not evidence of SHP early-warning success.

### 3. TS2Vec representations plus a small prediction head

TS2Vec learns contextual time-series representations through self-supervision.
The authors provide causal sliding inference that uses only preceding context.
[Paper](https://arxiv.org/abs/2106.10466);
[official implementation and causal example](https://github.com/zhihanyue/ts2vec).

**Project adaptation:** learn representations from the earlier telemetry without
fault labels, then train a small regularized classifier or event-level learner
on those representations. This may use the abundant unlabelled observations
more efficiently than learning a large supervised network from few events.

Each chronological fold must refit its encoder using only that fold's training
history. Default full-series encoding can expose later observations; use causal
inference and prefix-invariance tests. The original environment is old, so isolate
and test compatibility rather than downgrade the project's working environment.
This is a higher-effort follow-up, not the first experiment.

### 4. Multiple-instance learning, including attention pooling or WETAS

Multiple-instance learning attaches a label to a collection of instances rather
than every individual instance. Attention pooling can learn which instances
contribute to a collection's classification. WETAS addresses temporal anomaly
segmentation with weak instance-level labels.
[Attention-based MIL, ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html);
[WETAS, ICCV 2021](https://arxiv.org/abs/2108.06816).

**Project adaptation:** use an event's precursor interval as a positive collection
of short windows; use selected-normal intervals as controls. Learn a sparse
pattern or pooling rule without demanding that all precursor windows look alike.
The assumption that each positive collection contains a measurable precursor
remains unverified. Start small: a deep attention network has few independent
positive groups to learn from. WETAS detects anomalous segments; it is not already
a future-fault forecaster. A causal prediction adaptation needs its own evidence.

### 5. Conditional normal-behavior models

Hydropower work models expected component measurements given working-condition
variables, then monitors deviations from those relationships.
[Kaplan-turbine primary study](https://repositorio.comillas.edu/xmlui/handle/11531/28605).

**Project adaptation:** predict each sensor from other available sensors and past
context, then evaluate persistent conditional residuals for pre-fault association.
This might distinguish normal operating changes from broken sensor relationships.
It is distinct from simply reconstructing a temporal window or removing its mean.
The subsequent author-code review establishes the channel groups and V5/V6
meanings, but units, individual vibration orientations and operating modes remain
unconfirmed; inferred clusters are statistical regimes, not verified failure modes.
This is another viable existing-data experiment, with a weaker physical grounding
than the cited study.

### Foundation-model alternatives checked

**MOMENT:** open pretrained time-series representations with limited-supervision
evaluation; a frozen encoder and small head is a reasonable later comparator.
Check source-data overlap, channel handling, masks and trailing-window inference
before interpreting transfer results.
[Paper](https://arxiv.org/abs/2402.03885);
[official code](https://github.com/moment-timeseries-foundation-model/moment).

**TabPFN:** a strong alternative tabular learner, including current checkpoints
intended for distribution-shift scenarios. It still depends on useful features
and labels; it cannot recover temporal structure discarded before its input.
Consider it for an event-balanced feature table, not as evidence that the
underlying forecasting problem is solved. The checked TabPFN-3 model card has
specific research/production licensing conditions.
[Original Nature paper](https://doi.org/10.1038/s41586-024-08328-6);
[official model card](https://huggingface.co/Prior-Labs/tabpfn_3).

**Chronos-2:** supports multivariate and covariate-informed forecasting. It could
forecast expected sensor trajectories, but a sensor forecast is not a fault
probability. It would still need a validated mapping from forecast deviations to
future fault events, so it is lower priority here.
[Amazon research description](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting);
[official model card](https://huggingface.co/amazon/chronos-2).

I would not begin with another large Transformer trained from scratch, synthetic
fault generation, or ordinary remaining-useful-life regression: the effective
number of independent events is small, and repair/run-to-failure histories are
not established. These are project-specific priorities, not claims that those
techniques are generally ineffective.

## Concrete next experiment

Update after the author's repository review: first compare genuine EIF with
ordinary IF under the same chronological protocol, then evaluate operating-context
residuals using the now-supported V5/V6 meanings. See
[the source review](shp-source-code-review.md). The sequence-model plan below
remains a follow-up, not the immediate first experiment.

Implement a small **event-aware precursor dictionary**, alongside a fixed
MiniROCKET baseline. Keep this experiment separate from the rejected runs.

1. Use the verified chronological source union. Generate trailing windows of
   7, 13 and 25 observed readings for the dictionary; require 150-450-second
   intervals and retain actual elapsed durations. No gap filling or concatenation
   across outages. The MiniROCKET comparator initially uses 13 readings only.
2. Search shapes within training precursor windows. Require support in at least
   two different non-overlapping event groups, excluding the candidate's own
   group during scoring. Contrast with earlier selected-normal controls; rank
   and freeze a small dictionary using training-only evidence.
3. Keep normalized shape and amplitude evidence distinguishable. Flat windows
   need a defined treatment; normalization must not turn measurement noise into
   a strong pattern. Never interpret statistical motifs as physical fault modes.
4. Produce a causal similarity score at the current endpoint. Event timestamps
   determine historical training membership only; inference never receives the
   future event time. Do not backdate the decision to the beginning of a motif.
5. Calibrate the alert threshold on the next month's selected-normal observations,
   retaining the six-hour warning target, 2% point-alert budget and three-reading
   persistence. Evaluate subsequent months with all coverage exclusions retained.
6. Compare native and shared endpoints against the existing baseline where its
   chronology permits. Do not apply the frozen pre-February-trained LSTM to
   earlier months as though it were an out-of-sample comparator.
7. Report event-group detections, first-warning lead times, nominal alerts and
   episodes, plus the actual matched motifs. Favor patterns recurring before
   multiple faults, rather than a single impressive example.

This is an implementation brief, not a completed preregistration. Before running,
freeze candidate-bank size, dictionary cap, distance/normalization details,
tie-breaking, classifier settings, weighting and stopping criteria in a versioned
configuration. Do not search until a positive result appears. All inspected
months remain development data, including further earlier-history backtests;
chronological splits prevent look-ahead but do not erase repeated-selection bias.
Issue #21 still owns independent confirmation.

## Same-dataset evidence and practical status

The source SHP paper used Extended Isolation Forest with a publisher-defined
12-hour pre-fault split. Its result is a useful reference, but it does not validate
our chronological six-hour forecasting task. Copying its evaluation would answer
a different question.
[Original paper](https://www.wisdomlib.org/uploads/journals/mdpi-sust/2020-volume-12-issue-16--2071-1050-12-16-6421-.pdf).

Local checks found PyTorch available; aeon, sktime, numba, STUMPY, transformers
and TabPFN are not installed. No dependencies or model weights were downloaded,
no restricted telemetry was uploaded, and no experiment results were overwritten.
The next implementation should use an isolated environment for new dependencies.

GitHub issue #20 was attempted through the public web interface but could not be
read. The previous credential-access request was declined. This document uses
verified local experiment artifacts, does not assert current remote issue state,
and does not claim an issue update or publication. It is a local research finding
for the predictive-modeling work associated with #19, #20 and #21.
