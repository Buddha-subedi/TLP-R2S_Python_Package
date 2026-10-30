# From this ZIP to a public Python package

## A. Edit and test locally

1. Unzip the project; open its root in VS Code or terminal.
2. Install Python 3.11 (recommended first target), Git and an editor.
3. Create a virtual environment and install editable package:

   ```bash
   python -m venv .venv
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   # Linux/macOS: source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e '.[plot,dev]'
   python -m pytest -q
   ```

4. Run `python examples/demo_visualization.py` to generate **synthetic** maps without models.
5. Place your original models in `models/` and calibration NPZs in `cdf/`. Both directories are Git-ignored.
6. Run the real 2D inference example after replacing its placeholder input file with your 30 feature arrays.
7. Perform every step in `MIGRATION_CHECKLIST.md` and fix any mismatches.

## B. GitHub

1. Verify redistribution rights and remove private paths/data/secrets.
2. Add a LICENSE you are authorized to use, a CITATION.cff, contact, repository URL, DOI and complete author list.
3. `git init`; `git add .`; `git commit -m "Initial tlpr2s package"`.
4. Create a public GitHub repository and push. Enable GitHub Actions.
5. Provide stable, versioned model artifacts with checksums and license terms, or require users to supply their own model files. The ZIP does not include models.

## C. PyPI

1. Verify distribution name `tlpr2s` is available and you are permitted to use it.
2. `python -m pip install --upgrade build twine`.
3. `python -m build`.
4. `python -m twine check dist/*`.
5. Publish a test release to TestPyPI using its documented trusted publishing setup, then test installation in a fresh environment.
6. Configure PyPI trusted publishing (recommended) for GitHub Actions; publish a tagged release.
7. Test `python -m pip install tlpr2s` in a fresh environment and run inference with your model artifacts.

## D. Ongoing maintenance

Use semantic versioning, CHANGELOG, issues, automated tests, release tags, and dependency security updates. Verify scientific outputs before changing XGBoost or SciPy versions. Never silently change calibration, feature order, grid conventions or units.

**No payment is normally required** for a public GitHub repository, PyPI publishing, or standard public-repository GitHub Actions. Hosting large datasets/models or extra compute may have costs.
