"""Reproduce the reported numeric-Boolean test F1 for LR and SVM.

This loads the CSV afresh, so notebook cell execution order cannot affect it.
"""

from pathlib import Path
import warnings

import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


def main():
    csv = Path(__file__).resolve().parent.parent / "credit_card_fraud_2026.csv"
    X = pd.read_csv(csv).set_index("transaction_id")
    y = X.pop("is_fraud")
    # Capture the three lists BEFORE converting Booleans to integers.
    continuous = X.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    boolean = X.select_dtypes(include="bool").columns.tolist()
    assert (len(continuous), len(categorical), len(boolean)) == (13, 5, 6)
    X[boolean] = X[boolean].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Output order: 13 continuous columns, 5 category codes, 6 Boolean columns.
    pre = ColumnTransformer([
        ("num", StandardScaler(), continuous),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
        ("bool", "passthrough", boolean),
    ])
    continuous_indices = list(range(13))
    categorical_indices = list(range(13, 18))
    boolean_indices = list(range(18, 24))
    post = ColumnTransformer([
        ("num", StandardScaler(), continuous_indices),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_indices),
        ("bool", "passthrough", boolean_indices),
    ])
    for name, model in {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42),
    }.items():
        pipeline = Pipeline([
            ("pre_smote", clone(pre)),
            ("smote", SMOTENC(categorical_features=categorical_indices, random_state=42)),
            ("post_smote", clone(post)),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        print(f"{name}: test F1 = {f1_score(y_test, pipeline.predict(X_test)):.6f}")


if __name__ == "__main__":
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        main()
