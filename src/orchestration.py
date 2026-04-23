"""Simple model orchestrator (BrainNet) to load multiple models and run predictions.
Files stored in data/models/ may be raw sklearn objects or dicts with keys: model, scaler, feature_names.
"""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


class BrainNet:
    def __init__(self, model_dir=None, model_list=None):
        self.model_dir = Path(model_dir or os.getenv('MODEL_DIR', 'data/models'))
        if model_list is None:
            env_list = os.getenv('MODEL_LIST', '')
            model_list = [x.strip() for x in env_list.split(',') if x.strip()]
        self.models = {}
        for m in model_list:
            path = Path(m)
            if not path.is_absolute():
                path = self.model_dir / m
            if path.exists():
                data = joblib.load(str(path))
                if isinstance(data, dict) and 'model' in data:
                    self.models[path.name] = data
                else:
                    self.models[path.name] = {'model': data, 'scaler': None, 'feature_names': None}

    def predict_all(self, X):
        """Run each model on X (pandas.DataFrame or array). Returns dict of results."""
        results = {}
        for name, meta in self.models.items():
            model = meta.get('model')
            scaler = meta.get('scaler')
            feature_names = meta.get('feature_names')
            X_input = X
            if feature_names and isinstance(X, pd.DataFrame):
                cols = [c for c in feature_names if c in X.columns]
                X_input = X[cols]
            try:
                if scaler is not None:
                    Xs = scaler.transform(X_input)
                else:
                    Xs = getattr(X_input, 'values', X_input)
            except Exception:
                Xs = getattr(X_input, 'values', X_input)

            try:
                preds = model.predict(Xs)
            except Exception:
                preds = None
            probs = None
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(Xs)
            except Exception:
                probs = None
            results[name] = {'preds': preds, 'probs': probs}
        return results

    def ensemble_predict_classification(self, X):
        """Return (preds, probs_avg) where preds is 0/1 numpy array and probs_avg is averaged probability if available."""
        results = self.predict_all(X)
        probs_list = []
        preds_list = []
        for out in results.values():
            if out['probs'] is not None:
                arr = out['probs']
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    p = arr[:, 1]
                else:
                    p = np.asarray(arr)
                probs_list.append(p)
            elif out['preds'] is not None:
                preds_list.append(np.asarray(out['preds']))
        if probs_list:
            stacked = np.vstack(probs_list)
            avg = np.mean(stacked, axis=0)
            return (avg >= 0.5).astype(int), avg
        if preds_list:
            stacked = np.vstack(preds_list)
            vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, stacked)
            return vote, None
        return None, None
