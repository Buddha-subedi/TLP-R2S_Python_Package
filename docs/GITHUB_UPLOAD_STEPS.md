# Upload tlpr2s to GitHub

## Before uploading

- The five pretrained model and CDF files are **already hosted** at https://huggingface.co/BuddhaSubedi/TLP-R2S. Do **not** upload their 6.43 GB again to GitHub.
- This ZIP contains no trained model artifacts. `.gitignore` excludes their filenames.
- Confirm permission to release the source code and model artifacts. Replace `LICENSE-TO-SELECT.txt` with an approved LICENSE file, and update CITATION.cff and the author metadata before an official release.
- Verify actual orbit inference and original-pipeline numerical equivalence before claiming the package is validated.

## GitHub upload using Git (recommended)

1. Extract the ZIP, open a terminal **inside** the `tlpr2s` directory (where `pyproject.toml` is).
2. Create an empty public GitHub repository called `tlpr2s` without generating a README.
3. Run:

```bash
git init
git add .
git status
git commit -m "Initial tlpr2s package with Hugging Face model hosting"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/tlpr2s.git
git push -u origin main
```

Replace `YOUR_USERNAME`. Do not push the ZIP itself as your only repository file; push the extracted source tree.

## Local installation and smoke test

```bash
python -m pip install -e ".[all,dev]"
python -m pytest -q
tlpr2s download-models
```

The last command downloads approximately 6.43 GB and may require extra space for XGBoost loading. Use `--revision` with a fixed Hugging Face commit SHA when validating a release.

## Working with an offline MSI/HPC node

Download the artifacts on an internet-connected login node, or copy them into a directory containing all five original files, then set `TLPR2S_MODEL_DIR=/path/to/models` before running jobs. No network is needed on compute nodes if the files are local.

## Publish to PyPI later

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

PyPI publication is a separate, deliberate step. `pip install tlpr2s` will only work after an actual PyPI release under that name.
