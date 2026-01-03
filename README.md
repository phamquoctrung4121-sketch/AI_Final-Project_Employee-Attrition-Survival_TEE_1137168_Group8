**AI_Final-Project_Employee-Attrition-Survival_TEE_1137168_Group8**

This project develops a reproducible Python pipeline to predict employee attrition (IBM HR Analytics), focusing on estimating time-to-attrition and identifying key drivers. The dual approach combines Survival Analysis and Machine Learning to balance interpretability and accuracy. Specifically, we use Kaplan-Meier for data exploration, the Cox Proportional Hazards model as an interpretable baseline, and Random Survival Forest to capture non-linear effects. In parallel, an XGBoost classifier is trained and SHAP is applied to explain feature importance at both the global and individual levels. The outputs include not only risk probabilities for 3/6/12-month horizons but also survival curves, logloss curves for overfitting monitoring, and SHAP visualizations, providing a clear, data-driven foundation for HR decision-making.

**1) Project Description**
We model employee attrition as a time-to-event problem using survival analysis.
- Dataset: IBM HR Analytics (1,470 employees; 35 variables)
- Survival formulation:
  - event = 1 if Attrition = "Yes", else 0 (censored)
  - duration = YearsAtCompany × 12 (months)
- Pipeline: EDA/Kaplan–Meier → CoxPH → Random Survival Forest → Risk table (3/6/12 months) → XGBoost + SHAP.

**2) Repository Structure**
- data/: input CSV (or dataset link)
- src/run_pipeline.py: end-to-end pipeline
- results/: generated plots, tables, metrics
- assets/: slides / project materials

**3) Environment Setup**

Option A: conda
```bash
conda create -n ai_hr_survival python=3.11 -y
conda activate ai_hr_survival
conda install -c conda-forge numpy pandas matplotlib scikit-learn lifelines scikit-survival xgboost shap -y
```

Option B: pip
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**4) Reproduce Results (One Command)**
```bash
python src/run_pipeline.py --data_path data/hr_data.csv --out_dir results --seed 42
```

**5) Outputs**
- Kaplan–Meier plots: results/km_overtime.png, results/km_department.png
- Cox summary: results/cox_summary.csv
- RSF risk table (3/6/12 months): results/employee_risk_table_3_6_12m.csv
- XGBoost overfitting curve: results/xgb_logloss_overfitting.png
- SHAP summary: results/shap_summary.png
- Metrics summary: results/metrics.json

**6) Results Summary (fill from results/metrics.json)**
- Cox C-index (train/test): ...
- RSF C-index (train/test): ...
- Notes: AUC(t)/Brier may be unstable under censoring; we rely on C-index as primary.

**7) Notes / Limitations**
- IBM dataset is demo/synthetic; may differ from real HR systems.
- Time-dependent AUC/Brier can be unstable due to censoring at certain horizons.
