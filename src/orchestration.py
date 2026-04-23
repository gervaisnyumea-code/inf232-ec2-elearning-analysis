"""Simple model orchestrator (BrainNet) to load multiple models and run predictions.

This extended orchestrator supports:
- loading models saved as raw sklearn objects or dicts with keys: model, scaler, feature_names
- dynamic registration/unregistration of models
- weighted ensemble prediction for classification
- saving ensemble predictions to CSV
"""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import logging
import time
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class BrainNet:
    def __init__(self, model_dir: Optional[str] = None, model_list: Optional[list] = None, auto_load: bool = True):
        self.model_dir = Path(model_dir or os.getenv('MODEL_DIR', 'data/models'))
        self.models: Dict[str, Dict] = {}
        self.weights: Optional[Dict[str, float]] = None

        # Resolve model list from env if not provided
        if model_list is None:
            env_list = os.getenv('MODEL_LIST', '')
            model_list = [x.strip() for x in env_list.split(',') if x.strip()]

        # Try to load persisted weights if present
        self.load_weights()

        if auto_load:
            self.load_models(model_list)

    def _load_path(self, path: Path) -> Optional[Dict]:
        try:
            data = joblib.load(str(path))
            if isinstance(data, dict) and 'model' in data:
                model = data.get('model')
                scaler = data.get('scaler', None)
                feature_names = data.get('feature_names', None)
            else:
                model = data
                scaler = None
                feature_names = None
            return {'model': model, 'scaler': scaler, 'feature_names': feature_names, 'path': str(path)}
        except Exception as e:
            logger.exception('Failed to load model %s: %s', path, e)
            return None

    def load_models(self, model_list: Optional[list] = None, pattern: str = '*_meta.pkl') -> None:
        """Load models from provided list or from model_dir.

        If model_list is provided, load those file paths; otherwise load files matching pattern
        and then any remaining .pkl files.
        """
        if model_list:
            for m in model_list:
                p = Path(m)
                if not p.is_absolute():
                    p = self.model_dir / m
                if p.exists():
                    meta = self._load_path(p)
                    if meta:
                        self.models[p.name] = meta
        else:
            # load *_meta.pkl first (preferred), then others
            if self.model_dir.exists():
                for p in sorted(self.model_dir.glob(pattern)):
                    meta = self._load_path(p)
                    if meta:
                        self.models[p.name] = meta
                for p in sorted(self.model_dir.glob('*.pkl')):
                    if p.name.endswith('_meta.pkl'):
                        continue
                    # ignore obvious non-model files like scaler.pkl saved alone
                    if p.name.lower().startswith('scaler'):
                        continue
                    meta = self._load_path(p)
                    if meta:
                        self.models[p.name] = meta

    def add_model(self, name: str, model, scaler=None, feature_names: Optional[list] = None, overwrite: bool = False) -> None:
        if name in self.models and not overwrite:
            raise ValueError(f"Model '{name}' already registered. Use overwrite=True to replace.")
        self.models[name] = {'model': model, 'scaler': scaler, 'feature_names': feature_names, 'path': None}

    def remove_model(self, name: str) -> None:
        self.models.pop(name, None)

    def get_model_names(self) -> list:
        return list(self.models.keys())

    def set_weights(self, weights: Dict[str, float], persist: bool = True, filename: str = 'model_weights.json') -> None:
        """Set model weights for ensemble. Optionally persist to model_dir/filename."""
        self.weights = dict(weights or {})
        if persist:
            try:
                p = Path(self.model_dir) / filename
                p.parent.mkdir(parents=True, exist_ok=True)
                import json
                with open(p, 'w', encoding='utf-8') as fh:
                    json.dump(self.weights, fh)
            except Exception:
                logger.exception('Failed to persist weights')

    def load_weights(self, filename: str = 'model_weights.json') -> None:
        try:
            p = Path(self.model_dir) / filename
            if p.exists():
                import json
                with open(p, 'r', encoding='utf-8') as fh:
                    self.weights = json.load(fh)
        except Exception:
            self.weights = None

    def _prepare_input(self, X, feature_names: Optional[list]):
        if isinstance(X, pd.DataFrame):
            if feature_names:
                cols = [c for c in feature_names if c in X.columns]
                return X[cols]
            return X
        return X

    def predict_all(self, X):
        """Run each registered model on X (pandas.DataFrame or array). Returns dict of results."""
        results = {}
        for name, meta in self.models.items():
            model = meta.get('model')
            scaler = meta.get('scaler')
            feature_names = meta.get('feature_names')
            X_input = self._prepare_input(X, feature_names)

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

    def ensemble_predict_classification(self, X, weights: Optional[Dict[str, float]] = None, threshold: float = 0.5):
        """Weighted ensemble for classification models.

        Returns (preds, probs_avg) where preds is a 0/1 numpy array and probs_avg is the averaged
        probability vector if any model provides probabilities.
        """
        # Prefer explicit weights argument; otherwise try stored self.weights
        if weights is None:
            weights = getattr(self, 'weights', None)

        results = self.predict_all(X)
        probs_list = []  # tuples (name, pvec, weight)
        preds_list = []

        for name, out in results.items():
            p = out.get('probs', None)
            if p is not None:
                arr = np.asarray(p)
                if arr.ndim == 2:
                    pvec = arr[:, 1]
                else:
                    pvec = arr.ravel()
                w = 1.0 if (weights is None or name not in weights) else float(weights[name])
                probs_list.append((name, pvec, w))
            elif out.get('preds') is not None:
                preds_list.append((name, np.asarray(out['preds'])))

        if probs_list:
            weighted = np.vstack([pvec * w for (_, pvec, w) in probs_list])
            denom = max(sum(w for (_, _, w) in probs_list), 1e-12)
            avg = np.sum(weighted, axis=0) / denom
            preds = (avg >= threshold).astype(int)
            return preds, avg

        if preds_list:
            stacked = np.vstack([p for (_, p) in preds_list])
            vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, stacked)
            return vote, None

        return None, None

    def save_ensemble_predictions(self, X, out_csv: str = 'reports/ensemble_preds.csv') -> str:
        """Run all models on X and save a combined CSV with per-model preds/probas + ensemble."""
        res = self.predict_all(X)

        if isinstance(X, pd.DataFrame):
            df = X.reset_index(drop=True).copy()
        else:
            df = pd.DataFrame(getattr(X, 'tolist', lambda: X)())

        for name, out in res.items():
            preds = out.get('preds')
            probs = out.get('probs')
            df[f'{name}_preds'] = list(preds) if preds is not None else [None] * len(df)
            if probs is not None:
                arr = np.asarray(probs)
                if arr.ndim == 2:
                    df[f'{name}_proba'] = arr[:, 1]
                else:
                    df[f'{name}_proba'] = arr
            else:
                df[f'{name}_proba'] = [None] * len(df)

        preds_ens, proba_ens = self.ensemble_predict_classification(X)
        df['ensemble_pred'] = list(preds_ens) if preds_ens is not None else [None] * len(df)
        df['ensemble_proba'] = list(proba_ens) if proba_ens is not None else [None] * len(df)

        outp = Path(out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
        return str(outp)

    def run_on_live_stream(self, df_live: pd.DataFrame, out_csv: Optional[str] = None) -> Dict:
        """Run ensemble on a live-data snapshot (DataFrame) and optionally save results.

        Returns the dict with 'preds' and 'probs' (if available) and path if saved.
        """
        preds, probs = self.ensemble_predict_classification(df_live)
        result = {'preds': preds, 'probs': probs}
        saved = None
        if out_csv:
            saved = self.save_ensemble_predictions(df_live, out_csv=out_csv)
            result['saved_path'] = saved
        return result

    def __repr__(self):
        return f"<BrainNet models={len(self.models)}>"
