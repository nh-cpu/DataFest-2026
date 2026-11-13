import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from warranty_model import *


class WarrantyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.audit = prepare_data(Path(__file__).parent)

    def test_reconciliation_and_splits(self):
        self.assertTrue(self.data[ID].is_unique)
        claims = pd.read_excel(Path(__file__).parent / FILES['warranty'])
        self.assertAlmostEqual(self.data[TARGET].sum(), claims['Warranty Cost'].sum())
        splits = make_splits(self.data)
        pd.testing.assert_frame_equal(splits, make_splits(self.data.sample(frac=1, random_state=1)))
        train, test, cv = partition(self.data, splits)
        self.assertFalse(set(train.index) & set(test.index))
        self.assertEqual(sum(len(v) for _, v in cv), len(train))
        for t, v in cv:
            self.assertFalse(set(t) & set(v))

    def test_missing_unknown_leakage_and_reload(self):
        frame = self.data.head(100).set_index(ID)
        X = frame[FEATURES].copy()
        X.loc[X.index[:5], NUMERIC[0]] = np.nan
        model = build_pipeline(RandomForestRegressor(n_estimators=5, random_state=42)).fit(X, frame[TARGET])
        probe = X.head(3).copy()
        probe[CATEGORICAL[0]] = 'never observed category'
        probe[NUMERIC[0]] = np.nan
        first = model.predict(probe)
        probe['Warranty Cost'] = 1e12
        probe['Days To Failure'] = 99999
        np.testing.assert_array_equal(first, model.predict(probe))
        self.assertTrue(np.isfinite(first).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(model, path)
            np.testing.assert_array_equal(first, joblib.load(path).predict(probe))

    def test_metrics_alignment(self):
        frame = self.data.head(30).set_index(ID)
        model = build_pipeline(DummyRegressor()).fit(frame[FEATURES], frame[TARGET])
        metrics, predictions = evaluate_model(model, frame[FEATURES], frame[TARGET])
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        self.assertEqual(predictions[ID].tolist(), frame.index.tolist())
        with self.assertRaises(ValueError):
            evaluate_model(model, frame[FEATURES], frame[TARGET].iloc[::-1])

    def test_bad_split_rejected(self):
        split = make_splits(self.data)
        with self.assertRaises(ValueError):
            partition(self.data, split.iloc[:-1])

    def test_unknown_targets_and_repeated_claims(self):
        root = Path(__file__).parent
        tables = {key: pd.read_excel(root / filename) for key, filename in FILES.items()}
        unit = tables['warranty'].iloc[0][ID]
        extra = tables['warranty'].iloc[[0]].copy()
        tables['warranty'] = pd.concat([tables['warranty'], extra], ignore_index=True)
        unknown = tables['warranty'].iloc[1][ID]
        tables['warranty'].loc[tables['warranty'][ID].eq(unknown), 'Warranty Cost'] = np.nan
        def read(path, **kwargs):
            key = next(k for k, filename in FILES.items() if Path(path).name == filename)
            return tables[key].copy()
        with patch('warranty_model.pd.read_excel', side_effect=read):
            data, audit = prepare_data(root)
        self.assertNotIn(unknown, set(data[ID]))
        self.assertIn(unknown, audit['excluded_unknown_target_ids'])
        self.assertEqual(data.set_index(ID).loc[unit, TARGET], 2 * extra.iloc[0]['Warranty Cost'])

    def test_conflicting_feature_duplicates_rejected(self):
        root = Path(__file__).parent
        original = pd.read_excel
        def read(path, **kwargs):
            frame = original(path, **kwargs)
            if Path(path).name == FILES['rework']:
                duplicate = frame.iloc[[0]].copy()
                duplicate['Labor Hours'] = duplicate['Labor Hours'] + 1
                frame = pd.concat([frame, duplicate], ignore_index=True)
            return frame
        with patch('warranty_model.pd.read_excel', side_effect=read):
            with self.assertRaisesRegex(ValueError, 'conflicting duplicate'):
                prepare_data(root)

if __name__ == '__main__':
    unittest.main()
