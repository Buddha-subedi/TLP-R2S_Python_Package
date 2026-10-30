# Publishing tlpr2s on GitHub and PyPI

## A. Before publication
1. Check university/IP permissions for the research software and trained model/CDF artifacts. Choose a license (MIT/BSD/Apache only if permitted); do not imply a license without approval.
2. Add your authors, affiliation, scientific citation and release version to `pyproject.toml`, `CITATION.cff` and README.
3. Add five exact model/CDF artifacts under `src/tlpr2s/model_data/` **before** packaging, or intentionally choose external hosting and document the download.
4. Run real-orbit regression tests: phase identical, rain/snow within tolerances, original full NetCDF schema and scan times verified.
5. Check archive requirements with your destination NASA DAAC. A valid NetCDF file alone is not DAAC approval.

## B. Create GitHub repository
```bash
git init
git add .
git commit -m "Initial tlpr2s package"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/tlpr2s.git
git push -u origin main
```
Create the repository on GitHub first. Avoid committing raw data or proprietary model artifacts without permission. GitHub's web interface can create the repo.

## C. Install and test
```bash
python -m pip install -e ".[plot,io,dev]"
python -m pytest -q
python examples/demo_visualization.py
```
Run `examples/complete_workflow.py` after preparing actual `my_s4_arrays.npz` and supplying original model artifacts.

## D. Build and inspect wheel
```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
python - <<'PYINNER'
from pathlib import Path
import zipfile
for whl in Path('dist').glob('*.whl'):
    with zipfile.ZipFile(whl) as z:
        print('
'.join(x for x in z.namelist() if 'model_data/' in x))
PYINNER
```
If model files are absent from wheel, do not publish as a bundled-model release.

## E. Publish
Create a PyPI account and set up trusted publishing (recommended) or a scoped API token. Test on TestPyPI before production. Check that the name `tlpr2s` is available; do not assume it is reserved.

```bash
python -m twine upload --repository testpypi dist/*
# after testing, publish to production PyPI:
python -m twine upload dist/*
```
Then test in a fresh environment: `python -m pip install tlpr2s`.

## F. Annual-statistics caveat
Do not label an observed-swath mean as an unbiased annual accumulation. Annual accumulation requires temporal weighting/coverage assumptions or additional reconstruction.
