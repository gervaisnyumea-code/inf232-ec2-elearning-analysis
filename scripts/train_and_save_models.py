#!/usr/bin/env python3
"""Train regression and classification pipelines and save models with metadata.
Usage: python scripts/train_and_save_models.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_cleaning import full_pipeline, get_feature_matrix
from src.models import train_regression_pipeline, train_classification_pipeline


def main():
    print('Running full pipeline and training models...')
    df = full_pipeline()
    X, y_reg, y_clf = get_feature_matrix(df)

    reg_model, reg_metrics = train_regression_pipeline(X, y_reg)
    clf_model, clf_metrics = train_classification_pipeline(X, y_clf)

    # Save models (they include scaler and feature_names)
    Path('data/models').mkdir(parents=True, exist_ok=True)
    reg_model.save('data/models/regression_model.pkl')
    clf_model.save('data/models/classifier_model.pkl')

    print('Models trained and saved with metadata.')


if __name__ == '__main__':
    main()
