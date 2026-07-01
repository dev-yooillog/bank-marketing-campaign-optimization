import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from preprocessing import load_config


def load_data(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(cfg["data"]["final_path"])


def prepare_xy(df: pd.DataFrame, cfg: dict, drop_leakage: bool = False):
    target = cfg["target"]
    X = df.drop(columns=[target])
    y = df[target]

    if drop_leakage:
        leak_col = cfg.get("leakage_feature")
        if leak_col and leak_col in X.columns:
            X = X.drop(columns=[leak_col])

    return X, y


def get_models(random_state: int) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state
            ))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=3,      # 클래스 불균형 보정
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
    }


def train_and_evaluate(cfg: dict, drop_leakage: bool = False) -> dict:
    df = load_data(cfg)

    X, y = prepare_xy(df, cfg, drop_leakage=drop_leakage)

    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"],
        stratify=y
    )

    models = get_models(cfg["model"]["random_state"])
    cv = StratifiedKFold(n_splits=cfg["model"]["cv_folds"], shuffle=True,
                         random_state=cfg["model"]["random_state"])

    suffix = "_no_duration" if drop_leakage else "_with_duration"
    results = {}
    Path("outputs/models").mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\n[{name}{suffix}] 학습 중...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)

        results[f"{name}{suffix}"] = {
            "model": model,
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "X_test": X_test,
            "y_test": y_test
        }

        joblib.dump(model, f"outputs/models/{name}{suffix}.pkl")
        print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results


if __name__ == "__main__":
    cfg = load_config()

    print("\n= [Version A] duration 포함 =")
    results_with = train_and_evaluate(cfg, drop_leakage=False)

    print("\n= [Version B] duration 제외 (실전 타겟팅) =")
    results_without = train_and_evaluate(cfg, drop_leakage=True)