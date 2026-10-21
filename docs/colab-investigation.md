# SHP investigation in Colab

Use these two local files:

- `notebooks/investigate_shp_colab.ipynb`
- `artifacts/colab/hydropredict-investigation-2026-10-21.zip`

The second file is generated locally and ignored by Git. It contains the committed
source snapshot, the original checksum-verified development input package, baseline
score evidence and the frozen LSTM. It contains no final-test observations,
credentials, environment folders or live laptop connection.

## Run the notebook

1. Open [Google Colab](https://colab.research.google.com/) and upload
   `investigate_shp_colab.ipynb` as a notebook.
2. Use a standard Python CPU runtime. No paid runtime or GPU is needed.
3. Run cells in order. In the first upload prompt, choose the investigation ZIP
   above, not the older LSTM training ZIP. The notebook verifies its contents.
4. Review the preprocessing audit, coverage table, raw-versus-scaled plots and
   frozen LSTM evidence. `FAULT_INDEX = 1` selects the missed February 3 event;
   change the index to inspect another fault. This only changes visual diagnosis.
5. Run the fixed experiments and descriptive six-hour trend diagnosis. These call
   the committed repository implementations. They do not retrain the LSTM or tune
   its threshold. Existing output directories are preserved; use a new `RUN_NAME`
   or start a fresh runtime if rerunning the experiment cells.
6. Run the final export cell and save `investigation-v1-results.zip` (or the name
   matching your run). Download it before the runtime disconnects. It contains
   audit results, tables, figures, research candidate bundles, runtime metadata
   and checksums. No Drive authorisation is required.

The notebook is a 20-cell walkthrough with nine code cells. All nine computation
cells were executed successfully on an extracted portable package locally. The
96-test repository regression suite also passed. The Colab-specific upload,
dependency installation and download UI steps were not executed by the local check.

## Questions answered

- Does the saved scaler match the earlier weight-fitting subset rather than
  calibration or development observations?
- Are pre-fault measurements and timestamps preserved in the reconstructed input?
- Does scaling invert correctly, and do sequence endpoints retain channel order?
- Does CPU replay reproduce the stored scores and alert decisions?
- Which faults are adequately observed before and after sequence warm-up?
- Do missed events show unusual values, relationships or longer-duration changes?
- Does a candidate improve detection on identical timestamps without excessive
  nominal-reference alert burden?

Passing these checks establishes specific data-path contracts, not that every
possible preprocessing problem has been excluded. Signal semantics and source
timezone still need independent verification. Selected normal references, sparse
fault coverage, overlapping windows and prior use of development data prevent an
operational accuracy or false-alarm claim. The six-hour trends are descriptive,
not a validated detector. See [the study](signal-investigation-results.md).

## Rebuild a package after changing source

Commit your intended code and notebook changes first. The packager uses only
committed paths and an explicit evidence allowlist; unrelated untracked caches
are not included. Use a fresh output filename:

```powershell
.venv/Scripts/python.exe scripts/package_investigation.py --output artifacts/colab/hydropredict-investigation-new-run.zip
```

The existing original input ZIP and diagnostic folder must be available. Defaults
match the recorded local evidence. Use `--inputs` and `--diagnostics` to select
equivalent trusted artifacts elsewhere; matching checksums are required. The
package includes a source commit and member hashes. Keep the input package with
the downloaded result archive when archiving the investigation.
