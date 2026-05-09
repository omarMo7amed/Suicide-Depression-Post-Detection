import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MaxAbsScaler


def get_models() -> dict:

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            class_weight='balanced',
            random_state=42,
        ),
        "SVM (LinearSVC)": CalibratedClassifierCV(
            LinearSVC(
                max_iter=2000,
                C=1.0,
                class_weight='balanced',
                random_state=42,
            )
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            class_weight='balanced',
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42,
            algorithm='SAMME',
        ),
    }


# ── 2. Hyperparameter grids (for RandomizedSearchCV) ────────────────────────

PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 5.0, 10.0],
        "solver": ['lbfgs', 'saga'],
    },
    "SVM (LinearSVC)": {
        "estimator__C": [0.01, 0.1, 0.5, 1.0, 5.0],
    },
    "Decision Tree": {
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [15, 20, None],
        "min_samples_leaf": [1, 2, 4],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.5, 1.0, 1.5],
    },
}


# ── 3. Data splitting ────────────────────────────────────────────────────────

def split_data(X, y, test_size: float = 0.20, random_state: int = 42):
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


# ── 4. Cross-validation helper ───────────────────────────────────────────────

def cross_validate_model(model, X_train, y_train,
                         scoring: str = 'f1',
                         n_splits: int = 5) -> dict:

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring=scoring, n_jobs=-1)
    return {"mean": scores.mean(), "std": scores.std(), "scores": scores}


# ── 5. Optional hyperparameter tuning ────────────────────────────────────────

def tune_model(model, param_grid: dict,
               X_train, y_train,
               n_iter: int = 20,
               scoring: str = 'f1',
               n_splits: int = 5):

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV {scoring}: {search.best_score_:.4f}")
    return search.best_estimator_


# ── 6. Main training loop ────────────────────────────────────────────────────

def train_all_models(X_train, y_train,
                     tune: bool = False) -> dict:

    models = get_models()
    trained = {}

    for name, model in models.items():
        print(f"\n{'─'*55}")
        print(f"  Training: {name}")

        # CV score before fitting on full train set
        cv_result = cross_validate_model(model, X_train, y_train)
        print(f"  CV F1  : {cv_result['mean']:.4f} ± {cv_result['std']:.4f}")

        # Optional tuning
        if tune and name in PARAM_GRIDS and name != "K-Nearest Neighbors":
            print(f"  Tuning {name} …")
            model = tune_model(model, PARAM_GRIDS[name], X_train, y_train)

        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  ✓ {name} trained")

    return trained


# ── 7. Save / Load helpers ───────────────────────────────────────────────────

def save_models(trained_models: dict,
                directory: str = 'models/') -> None:
    os.makedirs(directory, exist_ok=True)
    for name, model in trained_models.items():
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        path = os.path.join(directory, f"{safe_name}.joblib")
        joblib.dump(model, path)
        print(f"Saved: {path}")


def load_model(name: str,
               directory: str = 'models/'):
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(directory, f"{safe_name}.joblib")
    return joblib.load(path)


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=100, random_state=42)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    models = train_all_models(X_tr, y_tr, tune=False)
    print("\nAll models trained successfully.")
