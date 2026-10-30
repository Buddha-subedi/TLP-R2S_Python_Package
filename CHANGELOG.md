# Changelog

## 0.1.0 (revised starter)
- Renamed package and CLI to `tlpr2s`.
- Default 0.25° grid uses 721 latitude nodes and 1440 periodic longitude nodes.
- Added optional bundled original XGBoost model loading.
- Preserved 2D-array input and orbit/annual plotting.

- Integrated original full DAAC-oriented products.py writer and public `DAAC_compatible_nc` API, requiring complete original SSMIS granule name and file for scan times.

## 0.2.0
- Connect official Hugging Face model repository with opt-in download at retriever initialization, cache, revision pinning, and offline overrides.
- Exclude large artifacts from wheels and Git.
- Add CLI `download-models` and integration tests.

- GitHub-ready CI workflow, upload instructions, and model provenance checklist added.
