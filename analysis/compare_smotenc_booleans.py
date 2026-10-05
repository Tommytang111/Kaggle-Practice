"""Compare Boolean handling in SMOTENC with otherwise identical processing.

Run from the repository root with an environment containing pandas, sklearn,
imbalanced-learn, and threadpoolctl. Results are saved beside this script.
The split and model parameters match credit_card_fraud2.ipynb, except SVC
probability calibration is disabled because only predicted labels are needed.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


def main():
    output_dir = Path(__file__).resolve().parent
    df = pd.read_csv(output_dir.parent / "credit_card_fraud_2026.csv")
    df = df.set_index("transaction_id")
    y = df.pop("is_fraud")
    numeric = df.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "string"]).columns.tolist()
    boolean = df.select_dtypes(include="bool").columns.tolist()
    # Identical 0/1 input and identical downstream Boolean representation.
    df[boolean] = df[boolean].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=42
    )
    numeric_indices = list(range(len(numeric)))
    categorical_indices = list(range(len(numeric), len(numeric) + len(categorical)))
    boolean_indices = list(range(len(numeric) + len(categorical), df.shape[1]))
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = [
        (str(fold), X_train.iloc[tr], X_train.iloc[va], y_train.iloc[tr], y_train.iloc[va])
        for fold, (tr, va) in enumerate(cv.split(X_train, y_train), start=1)
    ]
    splits.append(("test", X_train, X_test, y_train, y_test))
    results = []
    boolean_rates = []
    for fold, X_fit, X_eval, y_fit, y_eval in splits:
        pre = ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
            ("bool", "passthrough", boolean),
        ])
        a = pre.fit_transform(X_fit)
        b = pre.transform(X_eval)
        for treatment in ["categorical", "numeric"]:
            # This mask is the only treatment-specific code in the experiment.
            mask = categorical_indices + (boolean_indices if treatment == "categorical" else [])
            sampler = SMOTENC(categorical_features=mask, random_state=42)
            ar, yr = sampler.fit_resample(a, y_fit)
            synthetic = ar[len(a):, boolean_indices]
            for idx, feature in enumerate(boolean):
                boolean_rates.append({
                    "fold": fold, "treatment": treatment, "feature": feature,
                    "real_fraud_mean": float(X_fit.loc[y_fit == 1, feature].mean()),
                    "synthetic_mean": float(synthetic[:, idx].mean()),
                    "synthetic_fractional_share": float(np.mean(
                        (synthetic[:, idx] != 0) & (synthetic[:, idx] != 1)
                    )),
                })
            post = ColumnTransformer([
                ("num", StandardScaler(), numeric_indices),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_indices),
                ("bool", "passthrough", boolean_indices),
            ])
            ar = post.fit_transform(ar)
            be = post.transform(b)
            for name, estimator in models.items():
                model = clone(estimator)
                model.fit(ar, yr)
                pred = model.predict(be)
                result = {
                    "fold": fold, "treatment": treatment, "model": name,
                    "f1": f1_score(y_eval, pred, zero_division=0),
                    "precision": precision_score(y_eval, pred, zero_division=0),
                    "recall": recall_score(y_eval, pred, zero_division=0),
                }
                results.append(result)
                print(json.dumps(result), flush=True)
    scores = pd.DataFrame(results)
    scores.to_csv(output_dir / "smotenc_boolean_fold_scores.csv", index=False)
    pd.DataFrame(boolean_rates).to_csv(output_dir / "smotenc_boolean_synthetic_rates.csv", index=False)
    summaries = []
    for name in models:
        row = {"Model": name}
        for treatment in ["categorical", "numeric"]:
            cv_values = scores.loc[(scores.model == name) & (scores.treatment == treatment) & (scores.fold != "test"), "f1"]
            row[f"CV_F1_{treatment}"] = cv_values.mean()
            row[f"CV_SD_{treatment}"] = cv_values.std(ddof=1)
            row[f"Test_F1_{treatment}"] = scores.loc[(scores.model == name) & (scores.treatment == treatment) & (scores.fold == "test"), "f1"].iloc[0]
        row["CV_delta_numeric_minus_categorical"] = row["CV_F1_numeric"] - row["CV_F1_categorical"]
        row["Test_delta_numeric_minus_categorical"] = row["Test_F1_numeric"] - row["Test_F1_categorical"]
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "smotenc_boolean_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    # Keep BLAS and tree training bounded on the local machine.
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        main()
