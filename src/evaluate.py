import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from pathlib import Path

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

COLORS = {"primary": "#1B4FFF", "accent": "#F59E0B", "neg": "#374151"}


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"[{name}]")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(classification_report(y_test, y_pred))

    return {
        "name": name, "auc": auc, "ap": ap,
        "y_prob": y_prob, "y_pred": y_pred,
        "report": report
    }


def plot_roc_curves(eval_results: list, y_test, filename: str = "roc_curves.png") -> None:
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    for res in eval_results:
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{res['name']} (AUC={res['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve 비교")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{filename}", dpi=150)
    plt.close()
    print(f"[INFO] ROC curve 저장 완료: {filename}")


def plot_feature_importance(model_name: str, model, feature_names: list) -> None:
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    clf = model[-1] if hasattr(model, "__getitem__") else model

    if not hasattr(clf, "feature_importances_"):
        print(f"[SKIP] {model_name}: feature_importances_ 없음")
        return

    importances = pd.Series(clf.feature_importances_, index=feature_names)
    top20 = importances.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(8, 7))
    top20.plot(kind="barh", ax=ax, color=COLORS["primary"])
    ax.set_title(f"Feature Importance Top 20 ({model_name})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/feature_importance_{model_name}.png", dpi=150)
    plt.close()
    print(f"Feature importance 저장: {model_name}")

# 상위 k% 타겟팅 vs 전체 비교
def simulate_ab_test(y_test, y_prob, top_k_ratio: float = 0.30) -> None:
    df = pd.DataFrame({"actual": y_test.values, "prob": y_prob})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)

    n_total = len(df)
    n_top = int(n_total * top_k_ratio)

    control   = df                  # 전체
    treatment = df.iloc[:n_top]    # 상위 30%

    ctrl_cvr  = control["actual"].mean()
    treat_cvr = treatment["actual"].mean()
    cost_ratio = n_top / n_total

    print("\n[A/B 테스트 시뮬레이션]")
    print(f"  Control  전환율: {ctrl_cvr:.3f} (대상 {n_total}명)")
    print(f"  Treatment 전환율: {treat_cvr:.3f} (대상 {n_top}명, 상위 {top_k_ratio*100:.0f}%)")
    print(f"  접촉 비용 절감율: {(1 - cost_ratio)*100:.1f}%")
    print(f"  전환율 개선:     {(treat_cvr / ctrl_cvr - 1)*100:.1f}%")


def run_full_evaluation(cfg: dict, model_names: list, X_test, y_test, suffix: str) -> list:
    """버전(suffix)별로 모델들을 평가하고 ROC curve, importance, A/B 시뮬레이션까지 실행"""
    eval_results = []

    for base_name in model_names:
        full_name = f"{base_name}{suffix}"
        model = joblib.load(f"outputs/models/{full_name}.pkl")
        res = evaluate_model(full_name, model, X_test, y_test)
        eval_results.append(res)
        plot_feature_importance(full_name, model, list(X_test.columns))

    plot_roc_curves(eval_results, y_test, filename=f"roc_curves{suffix}.png")

    best = max(eval_results, key=lambda r: r["auc"])
    print(f"\n[Best model{suffix}] {best['name']} | AUC: {best['auc']:.4f}")
    simulate_ab_test(y_test, best["y_prob"],
                     top_k_ratio=cfg["thresholds"]["top_k_ratio"])

    return eval_results


if __name__ == "__main__":
    from preprocessing import load_config
    from train import load_data, prepare_xy
    from sklearn.model_selection import train_test_split

    cfg = load_config()
    df  = load_data(cfg)

    model_names = ["logistic_regression", "random_forest", "xgboost"]

    for drop_leakage, suffix in [(False, "_with_duration"), (True, "_no_duration")]:
        print(f"# 평가: {suffix}")

        X, y = prepare_xy(df, cfg, drop_leakage=drop_leakage)
        X = X.select_dtypes(include=[np.number])

        _, X_test, _, y_test = train_test_split(
            X, y,
            test_size=cfg["model"]["test_size"],
            random_state=cfg["model"]["random_state"],
            stratify=y
        )

        run_full_evaluation(cfg, model_names, X_test, y_test, suffix=suffix)