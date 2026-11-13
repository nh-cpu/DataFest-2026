# DataFest-2026: warranty cost prediction

An explained Random Forest notebook and reusable Python workflow for the Airedale Data Science Challenge.

## Run locally

Use Python 3.12. Create an environment, install dependencies, and select it as your notebook kernel:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m ipykernel install --user --name airedale --display-name "Airedale Python"
```

Open `random_forest_warranty.ipynb` in Jupyter or VS Code and run all cells from the project root.
Alternatively run `.venv/Scripts/python run_analysis.py --jobs 2`.
Run checks with `.venv/Scripts/python -m unittest test_warranty_model -v`.

## Google Colab

Upload the project folder, helper module, requirements, and four source workbooks to Google Drive.
Open the notebook in Colab and set `PROJECT_DIR` in its first code cell to the folder in Drive.
The notebook mounts Drive and installs `outputs/requirements-lock.txt` when available, otherwise `requirements.txt`.
Run all cells. No local Windows paths are required. Colab execution requires your own authenticated Drive session.

## Shared comparison contract

`outputs/modeling_dataset.csv` contains one eligible unit per row. `split_assignments.csv` holds the fixed 80/20 split and five training folds (test fold = -1). Join by `Serial Number`, never row position. `protocol.json` lists allowed predictors. Use `partition()` for identical fold positions and `evaluate_model()` with aligned serial-number indexes. Every model must fit preprocessing inside training folds.

Outputs include baseline/forest test metrics, CV results, predictions, audit and category mappings, source hashes, package versions and exact dependency lock, fitted pipeline, interpretation plots, and `results_summary.md`. Model binaries are ignored by Git but saved locally. Only load trusted joblib files.

The target is total **recorded** warranty cost. Missing targets are excluded, never zero-filled. The supplied claim-bearing population cannot establish accuracy for no-claim units or lifetime costs. Random splits do not measure future-production performance. Interpret feature importance as association, not causation. The holdout must not be used for further tuning.

No semantic label aliases are assumed without domain confirmation. See the exported mappings and `LABEL_ALIASES` for reviewed additions. Conflicting repeated feature records stop the workflow rather than being arbitrarily aggregated. Repeated warranty rows are treated as separate claim records because no claim ID is supplied.
