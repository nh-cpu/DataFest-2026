"""Reproducible, unit-level warranty cost modeling. See the notebook for interpretation."""
from pathlib import Path
import json
import platform
import importlib.metadata
import hashlib

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
ID = 'Serial Number'
TARGET = 'total_warranty_cost'
NUMERIC = ['Labor Hours', 'Material Cost', 'Rework Hours']
CATEGORICAL = ['Product Family', 'Facility', 'Controls Revision', 'Coil Supplier',
               'Rework Shift', 'Defect Category', 'Repair Type', 'Disposition Code',
               'EOLT Shift', 'Test Station', 'Failure Mode', 'Pass After Repair', 'Retested']
FEATURES = CATEGORICAL + NUMERIC
FILES = {
    'warranty': 'Competition_Dataset2_Warranty_Final.xlsx',
    'config': 'Competition_Dataset3_Product_Config_Final.xlsx',
    'rework': 'Competition_Dataset4_Rework_Final.xlsx',
    'eolt': 'Competition_Dataset5_EOLT_Final.xlsx',
}
# Add semantic synonyms only after domain confirmation. Formatting normalization
# is automatic and every observed original -> canonical value is exported.
LABEL_ALIASES = {}
REQUIRED = {
    'warranty': [ID, 'Warranty Cost'],
    'config': [ID, 'Product Family', 'Facility', 'Controls Revision', 'Coil Supplier'],
    'rework': [ID, 'Facility', 'Shift', 'Defect Category', 'Repair Type', 'Labor Hours', 'Material Cost', 'Disposition Code'],
    'eolt': [ID, 'Facility', 'Shift', 'Test Station', 'Failure Mode', 'Rework Hours', 'Pass After Repair', 'Retested'],
}


def save_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')


def prepare_data(data_dir):
    """Return eligible labeled units and a JSON-serializable data audit."""
    tables = {}
    audit = {'sources': {}, 'label_mappings': {}, 'facility_discrepancies': {}}
    for name, filename in FILES.items():
        path = Path(data_dir) / filename
        frame = pd.read_excel(path, dtype={ID: str})
        frame.columns = frame.columns.str.strip()
        missing = set(REQUIRED[name]) - set(frame.columns)
        if missing:
            raise ValueError(f'{name}: missing columns {sorted(missing)}')
        frame = frame[REQUIRED[name]].copy()
        frame[ID] = frame[ID].astype('string').str.strip().str.upper()
        if frame[ID].isna().any() or frame[ID].eq('').any():
            raise ValueError(f'{name}: missing unit identifier')
        audit['sources'][name] = {'rows': len(frame), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for col in frame.columns:
            if col == ID:
                continue
            if col in NUMERIC + ['Warranty Cost']:
                frame[col] = pd.to_numeric(frame[col], errors='raise').astype(float)
                if np.isinf(frame[col]).any() or frame[col].lt(0).any():
                    raise ValueError(f'{name}/{col}: negative or infinite values require review')
            else:
                raw = frame[col].copy()
                clean = raw.astype('string').str.strip().str.replace(r'\s+', ' ', regex=True).str.casefold()
                clean = clean.replace('', pd.NA).replace(LABEL_ALIASES.get(col, {}))
                audit['label_mappings'][f'{name}/{col}'] = {
                    str(a): str(b) for a, b in zip(raw, clean) if pd.notna(a)
                }
                frame[col] = clean.astype(object).where(clean.notna(), np.nan)
        if name != 'warranty':
            before = len(frame)
            frame = frame.drop_duplicates()
            audit['sources'][name]['exact_duplicates_removed'] = before - len(frame)
            if frame[ID].duplicated().any():
                raise ValueError(f'{name}: conflicting duplicate unit records')
        tables[name] = frame
    claims = tables['warranty']
    # Missing any claim amount makes that unit's total unknown, not a partial total.
    totals = claims.groupby(ID)['Warranty Cost'].agg(lambda s: s.sum() if s.notna().all() else np.nan).rename(TARGET)
    data = tables['config'].copy()
    roster = set(data[ID])
    for name, frame in tables.items():
        audit['sources'][name]['not_in_config'] = sorted(set(frame[ID]) - roster)
        audit['sources'][name]['config_without_record'] = sorted(roster - set(frame[ID]))
    for name in ['rework', 'eolt']:
        frame = tables[name]
        check = data[[ID, 'Facility']].merge(frame[[ID, 'Facility']], on=ID, suffixes=('_config', '_other'), validate='one_to_one')
        mismatch = check['Facility_config'].notna() & check['Facility_other'].notna() & check['Facility_config'].ne(check['Facility_other'])
        audit['facility_discrepancies'][name] = check.loc[mismatch].to_dict('records')
        frame = frame.drop(columns='Facility').rename(columns={'Shift': 'Rework Shift' if name == 'rework' else 'EOLT Shift'})
        data = data.merge(frame, on=ID, how='left', validate='one_to_one')
    data = data.merge(totals, on=ID, how='left', validate='one_to_one')
    audit['excluded_unknown_target_ids'] = data.loc[data[TARGET].isna(), ID].tolist()
    data = data.dropna(subset=[TARGET]).sort_values(ID).reset_index(drop=True)
    expected = claims.loc[claims[ID].isin(data[ID]), 'Warranty Cost'].sum()
    if not np.isclose(data[TARGET].sum(), expected, rtol=0, atol=0.01):
        raise ValueError('Target totals failed reconciliation')
    audit.update(eligible_units=len(data), target_total=float(data[TARGET].sum()),
                 zero_cost_units=int(data[TARGET].eq(0).sum()),
                 missing_predictors=data[FEATURES].isna().sum().to_dict(),
                 constant_predictors=[c for c in FEATURES if data[c].nunique(dropna=False) <= 1],
                 limitation='Recorded costs for represented units; no confirmed warranty horizon or no-claim population.')
    return data[[ID] + FEATURES + [TARGET]], audit


def make_splits(data):
    """Return persistent unit assignments; fold is the training validation fold."""
    ids = data[ID].sort_values().reset_index(drop=True)
    if ids.duplicated().any() or len(ids) < 10:
        raise ValueError('Need at least ten distinct eligible units')
    train, test = train_test_split(np.arange(len(ids)), test_size=0.2, random_state=SEED)
    result = pd.DataFrame({ID: ids, 'split': 'test', 'fold': -1})
    train = np.sort(train)
    result.loc[train, 'split'] = 'train'
    for fold, (_, val) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(train)):
        result.loc[train[val], 'fold'] = fold
    return result


def partition(data, splits):
    if data[ID].duplicated().any() or splits[ID].duplicated().any() or set(data[ID]) != set(splits[ID]):
        raise ValueError('Split assignments must cover each unit exactly once')
    joined = data.merge(splits, on=ID, validate='one_to_one').sort_values(ID).set_index(ID)
    if not joined['split'].isin(['train', 'test']).all():
        raise ValueError('Unknown split label')
    train, test = joined.loc[joined.split.eq('train')], joined.loc[joined.split.eq('test')]
    if len(test) < 2 or set(train.fold) != set(range(5)) or not test.fold.eq(-1).all():
        raise ValueError('Invalid test or cross-validation assignments')
    cv = [(np.flatnonzero(train.fold.to_numpy() != f), np.flatnonzero(train.fold.to_numpy() == f)) for f in range(5)]
    return train, test, cv


def build_pipeline(estimator=None):
    numeric = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    category = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing', keep_empty_features=True)),
                         ('encode', OneHotEncoder(handle_unknown='ignore'))])
    preprocess = ColumnTransformer([('numeric', numeric, NUMERIC), ('category', category, CATEGORICAL)], remainder='drop')
    if estimator is None:
        estimator = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=1, criterion='squared_error')
    return Pipeline([('preprocess', preprocess), ('model', estimator)])


def train_model(data, splits, n_jobs=2):
    """Tune on training folds only. Returns search and its refitted best pipeline."""
    train, _, cv = partition(data, splits)
    params = {'model__n_estimators': [300, 500, 800], 'model__max_depth': [None, 8, 16, 24],
              'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 5, 10],
              'model__max_features': [0.5, 0.75, 1.0]}
    search = RandomizedSearchCV(build_pipeline(), params, n_iter=30, scoring='neg_mean_absolute_error',
                                cv=cv, refit=True, n_jobs=n_jobs, random_state=SEED, error_score='raise')
    search.fit(train[FEATURES], train[TARGET])
    return search, search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """X_test index must be the unique unit identifier, aligned with y_test."""
    if not X_test.index.is_unique or not X_test.index.equals(y_test.index):
        raise ValueError('Unique aligned unit indexes required')
    pred = model.predict(X_test)
    if not np.isfinite(pred).all() or not np.isfinite(y_test).all():
        raise ValueError('Non-finite target or prediction')
    errors = np.abs(y_test.to_numpy() - pred)
    rng = np.random.default_rng(SEED)
    boots = np.array([rng.choice(errors, len(errors), replace=True).mean() for _ in range(1000)])
    metrics = {'mae': float(mean_absolute_error(y_test, pred)),
               'rmse': float(np.sqrt(mean_squared_error(y_test, pred))), 'r2': float(r2_score(y_test, pred)),
               'mae_ci95_low': float(np.quantile(boots, .025)), 'mae_ci95_high': float(np.quantile(boots, .975))}
    predictions = pd.DataFrame({ID: X_test.index, 'actual_cost': y_test.to_numpy(), 'predicted_cost': pred,
                                'residual': y_test.to_numpy() - pred, 'absolute_error': errors})
    return metrics, predictions


def fit_references(data, splits, n_jobs=2):
    train, _, cv = partition(data, splits)
    models, rows = {}, []
    for name, estimator in [('mean_baseline', DummyRegressor(strategy='mean')),
                            ('median_baseline', DummyRegressor(strategy='median')), ('initial_forest', None)]:
        model = build_pipeline(estimator)
        scores = cross_validate(model, train[FEATURES], train[TARGET], cv=cv, n_jobs=n_jobs,
                                scoring={'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}, error_score='raise')
        row = {'model': name}
        for metric in ['mae', 'rmse', 'r2']:
            values = scores[f'test_{metric}'] * (1 if metric == 'r2' else -1)
            row[f'cv_{metric}_mean'] = float(values.mean())
            row[f'cv_{metric}_std'] = float(values.std())
        rows.append(row)
        models[name] = model.fit(train[FEATURES], train[TARGET])
    return models, rows


def export_results(data, splits, audit, references, cv_rows, search, model, output_dir):
    """Final held-out evaluation. Do not use these results for further tuning."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='whitegrid', context='talk')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, test, _ = partition(data, splits)
    models = {**references, 'tuned_forest': model}
    results = []
    for name, fitted in models.items():
        metrics, predictions = evaluate_model(fitted, test[FEATURES], test[TARGET])
        results.append({'model': name, **metrics})
        predictions.to_csv(out / f'{name}_predictions.csv', index=False)
    metrics = pd.DataFrame(results)
    for baseline in ['mean_baseline', 'median_baseline']:
        score = metrics.loc[metrics.model.eq(baseline), 'mae'].iloc[0]
        metrics[f'mae_improvement_pct_vs_{baseline}'] = 100 * (1 - metrics.mae / score) if score else np.nan
    metrics.to_csv(out / 'test_metrics.csv', index=False)
    row = {'model': 'tuned_forest', 'cv_mae_mean': -float(search.best_score_),
           'cv_mae_std': float(search.cv_results_['std_test_score'][search.best_index_])}
    pd.DataFrame(cv_rows + [row]).to_csv(out / 'cv_summary.csv', index=False)
    pd.DataFrame(search.cv_results_).to_csv(out / 'cv_search_results.csv', index=False)
    data.to_csv(out / 'modeling_dataset.csv', index=False)
    splits.to_csv(out / 'split_assignments.csv', index=False)
    save_json(audit, out / 'data_audit.json')
    save_json(search.best_params_, out / 'best_parameters.json')
    save_json({'features': FEATURES, 'numeric': NUMERIC, 'categorical': CATEGORICAL,
               'target': TARGET, 'seed': SEED, 'test_fraction': .2, 'cv_folds': 5,
               'selection_metric': 'MAE', 'residual_definition': 'actual minus predicted'}, out / 'protocol.json')
    versions = {p: importlib.metadata.version(p) for p in ['pandas', 'numpy', 'openpyxl', 'scikit-learn', 'matplotlib', 'seaborn', 'joblib', 'nbformat', 'nbclient', 'ipykernel']}
    save_json({'python': platform.python_version(), 'packages': versions}, out / 'environment.json')
    (out / 'requirements-lock.txt').write_text(''.join(f'{p}=={v}\n' for p, v in versions.items()), encoding='utf-8')
    joblib.dump(model, out / 'random_forest.joblib')
    np.testing.assert_allclose(model.predict(test[FEATURES]), joblib.load(out / 'random_forest.joblib').predict(test[FEATURES]))
    _, pred = evaluate_model(model, test[FEATURES], test[TARGET])
    importance = permutation_importance(model, test[FEATURES], test[TARGET], scoring='neg_mean_absolute_error', n_repeats=20, random_state=SEED, n_jobs=2)
    imp = pd.DataFrame({'feature': FEATURES, 'mae_increase_mean': importance.importances_mean, 'mae_increase_std': importance.importances_std}).sort_values('mae_increase_mean')
    imp.to_csv(out / 'permutation_importance.csv', index=False)
    def finish(name):
        plt.tight_layout()
        plt.savefig(out / f'{name}.png', dpi=160, bbox_inches='tight')
        plt.close()
    plt.figure(figsize=(10, 6))
    sns.histplot(data[TARGET], bins=40)
    plt.xlabel('Recorded warranty cost ($)'); plt.ylabel('Units'); finish('cost_distribution')
    plt.figure(figsize=(8, 7))
    plt.scatter(pred.actual_cost, pred.predicted_cost, alpha=.5, s=20)
    bounds = [0, max(pred.actual_cost.max(), pred.predicted_cost.max())]
    plt.plot(bounds, bounds, '--', color='black')
    plt.xlabel('Actual cost ($)'); plt.ylabel('Predicted cost ($)'); finish('actual_vs_predicted')
    plt.figure(figsize=(10, 6))
    plt.scatter(pred.predicted_cost, pred.residual, alpha=.5, s=20); plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Predicted cost ($)'); plt.ylabel('Actual minus predicted ($)'); finish('residuals')
    plt.figure(figsize=(11, 9))
    plt.barh(imp.feature, imp.mae_increase_mean, xerr=imp.mae_increase_std)
    plt.xlabel('Increase in test MAE after shuffling ($)'); finish('permutation_importance')
    groups = pred.merge(test[CATEGORICAL], left_on=ID, right_index=True, validate='one_to_one')
    for col in ['Product Family', 'Facility']:
        summary = groups.groupby(col, dropna=False).agg(units=(ID, 'size'), mae=('absolute_error', 'mean'), mean_residual=('residual', 'mean')).reset_index()
        summary.to_csv(out / f'error_by_{col.lower().replace(" ", "_")}.csv', index=False)
        plt.figure(figsize=(10, 6))
        plt.bar(summary[col].fillna('Missing') + ' (n=' + summary.units.astype(str) + ')', summary.mae)
        plt.ylabel('Test MAE ($)'); plt.xticks(rotation=20); finish(f'error_by_{col.lower().replace(" ", "_")}')
    threshold = float(train[TARGET].quantile(.9))
    expensive = pred.loc[pred.actual_cost >= threshold]
    high = {'threshold_training_p90': threshold, 'test_units': len(expensive),
            'mean_underprediction_dollars': float(expensive.residual.mean()) if len(expensive) else None,
            'underpredicted_fraction': float(expensive.residual.gt(0).mean()) if len(expensive) else None}
    save_json(high, out / 'expensive_claims.json')
    best = metrics.loc[metrics.model.eq('tuned_forest')].iloc[0]
    report = f'''# Warranty model results

Test units: {len(test)}; training units: {len(train)}.
Random Forest MAE: ${best.mae:,.2f}; RMSE: ${best.rmse:,.2f}; R²: {best.r2:.4f}.
95% unit-bootstrap MAE interval: ${best.mae_ci95_low:,.2f} to ${best.mae_ci95_high:,.2f}.
This interval conditions on the fitted model and does not measure retraining uncertainty.

MAE improvement over mean baseline: {best.mae_improvement_pct_vs_mean_baseline:.2f}%.
MAE improvement over median baseline: {best.mae_improvement_pct_vs_median_baseline:.2f}%.
Negative improvement means the model is worse than that baseline.

Expensive claims use the training 90th percentile (${threshold:,.2f}); {len(expensive)} test units qualify.
Mean actual-minus-predicted cost for that group: {high['mean_underprediction_dollars']}.

These are recorded costs for represented units, with no confirmed observation horizon.
Random splits assess similar units, not future-production performance. No-claim units are not validated.
Permutation importance is predictive association, not causal evidence. Negative values are possible.
Use predictions to prioritize review; avoided costs require a separately evaluated intervention.
Keep the held-out set fixed and do not tune further using its diagnostics.
'''
    (out / 'results_summary.md').write_text(report, encoding='utf-8')
    return metrics
