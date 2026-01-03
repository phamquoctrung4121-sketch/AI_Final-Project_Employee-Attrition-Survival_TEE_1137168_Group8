import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score

import xgboost as xgb
import shap


def set_seed(seed: int):
    np.random.seed(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def make_y_struct(time_arr, event_arr):
    return np.array(
        [(bool(e), float(t)) for e, t in zip(event_arr, time_arr)],
        dtype=[("event", "?"), ("time", "<f8")],
    )


def plot_km(df: pd.DataFrame, col_name: str, out_path: Path):
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots()

    for label, grouped in df.groupby(col_name):
        kmf.fit(grouped["duration"], grouped["event"], label=str(label))
        kmf.plot_survival_function(ax=ax)

    ax.set_title(f"Survival curves by {col_name}")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_X_survival(df: pd.DataFrame) -> pd.DataFrame:
    """
    Survival features: MUST NOT include label leakage columns.
    Also drop YearsAtCompany because duration is defined from it (duration = YearsAtCompany * 12).
    """
    drop_cols = [
        "EmployeeNumber", "Attrition", "EmployeeCount", "Over18", "StandardHours",
        "event", "duration", "YearsAtCompany"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    data = df.drop(columns=drop_cols)

    cat_cols = data.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    return X


def build_X_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classification features: remove leakage columns (event/duration), but keep YearsAtCompany if you want
    tenure effects in SHAP (as your slide discusses).
    """
    drop_cols = [
        "EmployeeNumber", "Attrition", "EmployeeCount", "Over18", "StandardHours",
        "event", "duration"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    data = df.drop(columns=drop_cols)

    cat_cols = data.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    return X


def align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # Ensure train/test have identical columns after one-hot
    X_train_aligned, X_test_aligned = X_train.align(X_test, join="left", axis=1, fill_value=0)
    return X_train_aligned, X_test_aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/hr_data.csv")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # ========= 1) LOAD & DEFINE SURVIVAL TARGETS =========
    df = pd.read_csv(args.data_path)
    df["event"] = (df["Attrition"] == "Yes").astype(int)
    df["duration"] = df["YearsAtCompany"] * 12  # months (as in your slide)

    # One consistent split index for all models
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.3, random_state=args.seed, stratify=df["event"])

    # ========= 2) KAPLAN–MEIER PLOTS =========
    # (match your slide: overtime & department)
    plot_km(df, "OverTime", out_dir / "km_overtime.png")
    plot_km(df, "Department", out_dir / "km_department.png")

    # ========= 3) COX PROPORTIONAL HAZARDS =========
    X_surv_all = build_X_survival(df)
    X_surv_train = X_surv_all.iloc[idx_train].copy()
    X_surv_test = X_surv_all.iloc[idx_test].copy()
    X_surv_train, X_surv_test = align_columns(X_surv_train, X_surv_test)

    ytime_train = df.loc[idx_train, "duration"].values
    ytime_test = df.loc[idx_test, "duration"].values
    yevent_train = df.loc[idx_train, "event"].values
    yevent_test = df.loc[idx_test, "event"].values

    train_df = X_surv_train.copy()
    train_df["duration"] = ytime_train
    train_df["event"] = yevent_train

    test_df = X_surv_test.copy()
    test_df["duration"] = ytime_test
    test_df["event"] = yevent_test

    cph = CoxPHFitter()
    cph.fit(train_df, duration_col="duration", event_col="event")

    # Save Cox summary
    cph.summary.to_csv(out_dir / "cox_summary.csv", index=True)

    pred_haz_train = cph.predict_partial_hazard(train_df)
    pred_haz_test = cph.predict_partial_hazard(test_df)

    c_index_cox_train = concordance_index(train_df["duration"], -pred_haz_train.values.ravel(), train_df["event"])
    c_index_cox_test = concordance_index(test_df["duration"], -pred_haz_test.values.ravel(), test_df["event"])

    # ========= 4) RANDOM SURVIVAL FOREST =========
    y_train_struct = make_y_struct(ytime_train, yevent_train)
    y_test_struct = make_y_struct(ytime_test, yevent_test)

    rsf = RandomSurvivalForest(
        n_estimators=300,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=args.seed,
    )
    rsf.fit(X_surv_train, y_train_struct)

    # FIX: event indicator must be yevent (True = event happened)
    ci_rsf_train = concordance_index_censored(
        yevent_train.astype(bool), ytime_train, rsf.predict(X_surv_train)
    )[0]
    ci_rsf_test = concordance_index_censored(
        yevent_test.astype(bool), ytime_test, rsf.predict(X_surv_test)
    )[0]

    # AUC(t) & Brier (optional; may be unstable under censoring)
    times = np.array([12, 24, 36])
    auc_info = {"times": times.tolist(), "auc_t": None, "mean_auc": None, "ibs": None, "error": None}
    try:
        chf_test = rsf.predict_cumulative_hazard_function(X_surv_test)
        chf_test_array = np.row_stack([fn(times) for fn in chf_test])

        auc_scores, mean_auc = cumulative_dynamic_auc(y_train_struct, y_test_struct, chf_test_array, times)
        auc_info["auc_t"] = [float(x) for x in auc_scores]
        auc_info["mean_auc"] = float(mean_auc)

        surv_test = rsf.predict_survival_function(X_surv_test)
        surv_test_array = np.row_stack([fn(times) for fn in surv_test])
        ibs = integrated_brier_score(y_train_struct, y_test_struct, surv_test_array, times)
        auc_info["ibs"] = float(ibs)
    except ValueError as e:
        auc_info["error"] = str(e)

    # ========= 5) RISK TABLE 3 / 6 / 12 MONTHS =========
    horizons = np.array([3, 6, 12])
    surv_all = rsf.predict_survival_function(X_surv_all)
    surv_all_array = np.row_stack([fn(horizons) for fn in surv_all])
    risk_probs = 1 - surv_all_array

    def risk_tier(p):
        if p >= 0.6:
            return "High"
        elif p >= 0.3:
            return "Medium"
        else:
            return "Low"

    risk_df = pd.DataFrame(
        {
            "EmployeeID": df["EmployeeNumber"],
            "Prob_leave_3m": risk_probs[:, 0],
            "Prob_leave_6m": risk_probs[:, 1],
            "Prob_leave_12m": risk_probs[:, 2],
        }
    )
    risk_df["RiskTier_6m"] = risk_df["Prob_leave_6m"].apply(risk_tier)
    risk_df = risk_df.merge(
        df[["EmployeeNumber", "Department", "JobRole"]],
        left_on="EmployeeID",
        right_on="EmployeeNumber",
    ).drop(columns=["EmployeeNumber"])
    risk_df.to_csv(out_dir / "employee_risk_table_3_6_12m.csv", index=False)

    # ========= 6) XGBOOST + SHAP (Classification view + overfitting monitoring) =========
    X_clf_all = build_X_classification(df)
    X_clf_train = X_clf_all.iloc[idx_train].copy()
    X_clf_test = X_clf_all.iloc[idx_test].copy()
    X_clf_train, X_clf_test = align_columns(X_clf_train, X_clf_test)

    y_clf_train = df.loc[idx_train, "event"].values
    y_clf_test = df.loc[idx_test, "event"].values

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.seed,
    )

    eval_set = [(X_clf_train, y_clf_train), (X_clf_test, y_clf_test)]
    xgb_model.fit(X_clf_train, y_clf_train, eval_set=eval_set, verbose=False)

    results = xgb_model.evals_result()
    train_logloss = results["validation_0"]["logloss"]
    test_logloss = results["validation_1"]["logloss"]

    epochs = range(1, len(train_logloss) + 1)
    fig = plt.figure()
    plt.plot(epochs, train_logloss, label="Train logloss")
    plt.plot(epochs, test_logloss, label="Test logloss")
    plt.xlabel("Boosting rounds")
    plt.ylabel("Logloss")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "xgb_logloss_overfitting.png")
    plt.close(fig)

    # SHAP (use test set)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_clf_test)

    plt.figure()
    shap.summary_plot(shap_values, X_clf_test, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", bbox_inches="tight")
    plt.close()

    # ========= 7) SAVE METRICS =========
    metrics = {
        "data_rows": int(df.shape[0]),
        "data_cols": int(df.shape[1]),
        "attrition_yes": int(df["event"].sum()),
        "attrition_no": int((1 - df["event"]).sum()),
        "cox_c_index_train": float(c_index_cox_train),
        "cox_c_index_test": float(c_index_cox_test),
        "rsf_c_index_train": float(ci_rsf_train),
        "rsf_c_index_test": float(ci_rsf_test),
        "rsf_auc_brier_optional": auc_info,
        "outputs": {
            "km_overtime": "km_overtime.png",
            "km_department": "km_department.png",
            "cox_summary": "cox_summary.csv",
            "risk_table": "employee_risk_table_3_6_12m.csv",
            "xgb_logloss": "xgb_logloss_overfitting.png",
            "shap_summary": "shap_summary.png",
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Done. Outputs saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
