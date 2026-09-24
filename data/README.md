# Data workspace

- `sample/` contains small, synthetic, version-controlled examples.
- `raw/` contains immutable source files acquired from approved datasets.
- `processed/` contains reproducible outputs derived from raw data.
- `quarantine/` contains rejected records and validation reasons.
- `dataset_manifest.csv` records provenance and readiness.

Raw, processed, and quarantined contents are ignored by Git except for their
placeholder files. Do not put credentials, personal information, or restricted
plant data in the repository.

For a downloaded file, generate its checksum in PowerShell with:

```powershell
Get-FileHash -Algorithm SHA256 .\data\raw\FILE_NAME
```

Then add the retrieval date, licence, filename, and checksum to the manifest.

