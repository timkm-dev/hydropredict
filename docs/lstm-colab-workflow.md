# LSTM implementation and Colab workflow

14 October 2026. The compact LSTM, sequence builder, resumable training, export, CPU loading,
validation command and thin Colab notebook are implemented. **Full hosted GPU training and
fault-based LSTM performance remain pending until the owner runs the notebook.** Engineering
tests and the separate local smoke test do not establish scientific performance.

## The laptop and Colab connection

The laptop holds the source code. `scripts/package_colab.py` takes a committed snapshot and
the declared input CSVs and produces one ZIP. Upload that ZIP inside the Colab notebook.
Colab installs and executes the copied repository code on its hosted GPU, saves checkpoints
to your Drive and exports a ZIP containing the model and reports. Download and extract that
output locally. There is no live laptop connection, automatic synchronisation, tunnelling
or second implementation of the training algorithm in notebook cells.

Notebook: `notebooks/train_lstm_colab.ipynb`.
Initial input package: `artifacts/colab/hydropredict-lstm-input-2026-10-14.zip`.
Generated inputs and trained artifacts remain ignored by Git. The snapshot excludes local
environments, caches, Git internals, credentials and academic documents. Only committed
source under the selected project paths and explicitly selected data are included.

## Steps for the owner

1. Open [Google Colab](https://colab.research.google.com/) and use File / Upload notebook to
   open `train_lstm_colab.ipynb` from the project notebook directory.
2. Select Runtime / Change runtime type / GPU. GPU availability is variable; the notebook
   refuses full training if no hosted GPU is available. A local runtime would use the
   laptop and is not the intended execution environment.
3. Run the upload cell and select the single prepared input ZIP named above. It verifies
   code and data checksums, extracts a fresh workspace and installs the project.
4. Review the displayed initial configuration. Run the Drive connection cell and authorise
   access interactively. Keep `RUN_NAME` fixed for this experiment. Results and checkpoints
   are written under `MyDrive/HydroPredict/<RUN_NAME>`.
5. Run training. It stops at thirty epochs or after five normal early-stopping epochs
   without sufficient improvement, whichever comes first. No fault labels select the epoch
   or threshold. The notebook can resume a matching interrupted run.
6. Run the separate validation cell after the bundle is exported. It checks provenance and
   evaluates only the development interval, including original and model-score coverage.
7. Run the export cell and download `hydropredict-lstm-export.zip`. Keep its training and
   validation reports with the bundle. Extract it into a new local model folder, and return
   the archive or its location for comparison with the stored Isolation Forest results.

If Colab disconnects, start a fresh hosted runtime, upload the **same input ZIP**, use the
same run name and run the cells again. A matching checkpoint resumes the next epoch.
Changed input data, configuration or source provenance is rejected. Completed bundles and
existing validation outputs are preserved. Use a new run name for a different experiment;
do not overwrite one to hide its results. Paid Colab is not required by this implementation.

## Fixed initial experiment

`configs/lstm_initial.json` specifies seven readings, hidden size 32, latent size 16,
batch size 128, Adam learning rate 0.001, maximum thirty epochs, patience five, minimum
improvement 0.00001, calibration quantile 0.99 and seed 42. Encoder and decoder each have
one LSTM layer. A linear bottleneck compresses the encoder's final hidden state; its repeated
latent vector feeds the decoder, whose linear output reconstructs all six input channels.
There is no skip connection from the input to the decoder output.

Seven observations contain six nominal five-minute intervals, so they span about thirty
minutes. Endpoints retain the actual start and end timestamps. Allowed adjacent steps are
150 to 450 seconds; other steps reset the sequence. Actual durations therefore vary with
sampling. No filling, source-membership labels, engineered physical assumptions, countdowns
or fault labels enter the tensor. Scores are assigned to the last observed reading in each
backward-only window. Each partition starts fresh, losing its first six eligible context
readings per segment. A future streaming caller must retain preceding history.

From the original 30,807 chronological normal-reference observations, the first 80% remain
the normal development block and the last 20% remain calibration. Within the first block,
the first 90% fit weights and the final 10% select an epoch by reconstruction loss. The
scaler fits only the weight-fitting subset. This extra early-stopping holdout means the LSTM
uses fewer weight-fitting observations than the Isolation Forest; disclose that difference.
Overlapping windows never cross partition boundaries or invalid cadence gaps.

Training minimises mean squared reconstruction error and clips gradient norm at 1.0.
The lowest normal early-stopping loss determines the exported weights. The 99th percentile
of reconstruction scores on the separate normal calibration subset sets the threshold.
An alert requires score strictly greater than threshold. No persistence or cooldown is
added to the initial experiment, and 99th-percentile calibration is not a guaranteed
operational false-alert rate.

## Explanations and evaluation

The overall anomaly score is mean squared error across sequence steps and channels after
training-derived standardisation. Per-channel errors are returned alongside it. These are
reconstruction-error explanations, not SHAP values, causal attributions or verified physical
fault diagnoses. Channel meanings and physical plant timezone remain unresolved.

Only after training and calibration does `scripts/evaluate_lstm.py` load the separate
validation fault file. The package includes only validation fault timestamps, no final-test
telemetry. Evaluation retains the original six-hour warning, 80% slot-coverage and ten-minute
latest-observation rules. It reports original eligibility, adequately scored eligibility,
event detections, lead times, sequence exclusions and selected nominal-reference alert
burden. Nominal samples remain retrospectively publisher-selected; general fleet false-alert
frequency is not established. Validation has already been used for development, and a fresh
final evaluation protocol remains necessary.

## Export and local use

`bundle/` contains CPU weights in a state dictionary, scaler JSON, sequence and architecture
configuration, threshold, model version, environment versions, provenance and checksums.
Training history, timing, partition ranges and row/sequence accounting are kept in the run.
Checkpoints additionally contain optimizer state, best weights and early-stopping history.
The trainer compares exported CPU reconstruction errors with training-device errors on a
fixed calibration sample using rtol=0.0001 and atol=0.000001. A full GPU-to-CPU check is
pending until Colab execution; local CPU round trips have been verified.

Install CPU PyTorch locally, or use the updated Windows setup script:

```powershell
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[lstm]"
```

```python
from hydropredict.models.lstm import SHPLSTMAutoencoder
from hydropredict.data.preprocessing import load_telemetry_dataframe

model = SHPLSTMAutoencoder.load("models/my-colab-run/bundle")
scores = model.score(load_telemetry_dataframe("path/to/validated_measurements.csv"))
```

This inference call defaults to CPU and needs no Colab connection. Checksums are verified
before loading weights with `weights_only=True` and `map_location="cpu"`. Environment
versions are recorded; if a framework incompatibility occurs, use a compatible CPU PyTorch
version from the bundle metadata. No cross-version or hardware-identical timing guarantee
is claimed. Full quality checks require the LSTM extra; CI installs CPU PyTorch and runs
the model tests without a GPU.

## Rebuild the upload package

Commit source changes first, then use a new output name:

```powershell
python scripts/package_colab.py --preparation-run data/processed/runs/2026-10-13-verified --context-run models/continuous-context-2026-10-14-v2 --output artifacts/colab/my-new-input.zip
```

The package's manifest identifies the committed source and every input checksum. Windows
absolute paths are not required inside Colab. Upload a new snapshot when training code
changes. The tiny local command `scripts/train_lstm.py --smoke-test` explicitly uses CPU,
two epochs and limited normal subsets. Its bundle is labelled engineering-only and is
rejected by scientific validation, preventing smoke results from being mistaken for the
completed Colab experiment.

References: [PyTorch state-dictionary export](https://docs.pytorch.org/tutorials/beginner/saving_loading_models),
[Colab resource and persistence guidance](https://research.google.com/colaboratory/faq.html).
