**AI_Final-Project_Employee-Attrition-Survival_TEE_1137168_Group8**

**📌 Project Overview**

This project develops a reproducible Python pipeline to predict employee attrition (IBM HR Analytics), focusing on estimating time-to-attrition and identifying key drivers. The dual approach combines Survival Analysis and Machine Learning to balance interpretability and accuracy. Specifically, we use Kaplan-Meier for data exploration, the Cox Proportional Hazards model as an interpretable baseline, and Random Survival Forest to capture non-linear effects. In parallel, an XGBoost classifier is trained and SHAP is applied to explain feature importance at both the global and individual levels. The outputs include not only risk probabilities for 3/6/12-month horizons but also survival curves, logloss curves for overfitting monitoring, and SHAP visualizations, providing a clear, data-driven foundation for HR decision-making.

**🎯 Objectives & Methodology**

Objectives: Understand attrition causes and estimate resignation risk over time (3, 6, 12 months).

Methodology:

- Survival Analysis: Framing the problem as a "time-to-event" task.
 
-- event = 1 if Attrition == "Yes".
-- duration = YearsAtCompany × 12 (months).
-- Employees who haven't left are considered "right-censored".

- Core Models:

-- Kaplan-Meier: Exploratory analysis and survival comparison across groups (e.g., department, overtime status).
-- Cox Proportional Hazards (CoxPH): A linear, highly interpretable model used as a baseline.
-- Random Survival Forest (RSF): An ensemble model to capture non-linear effects and complex interactions.
-- XGBoost Classifier: A powerful classification model used in parallel for comparison.
-- Explainable AI (XAI): Using SHAP (SHapley Additive exPlanations) to explain feature importance at both global and individual levels.

**📁 Repository Structure**

├── data/                   # Folder for input data (CSV)

├── src/                    # Main source code

│   └── run_pipeline.py     # Script to run the entire pipeline

├── results/                # Output folder for plots, tables, metrics

├── assets/                 # Supplementary materials (slides, reports)

├── requirements.txt        # List of required Python libraries

└── README.md               # This guide


**⚙️ Environment Setup & Installation**

Option A: Using pip (Recommended)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```
Sample requirements.txt content:

- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- scikit-learn>=1.0.0
- lifelines>=0.27.0
- scikit-survival>=0.19.0
- xgboost>=1.5.0
- shap>=0.41.0

Option B: Using Conda

```bash
conda create -n ai_hr_survival python=3.11 -y
conda activate ai_hr_survival
conda install -c conda-forge numpy pandas matplotlib scikit-learn lifelines scikit-survival xgboost shap -y
```

**4🚀 Running the Pipeline & Reproducing Results**

To execute the entire analysis pipeline from start to finish with a single command:

```bash
python src/run_pipeline.py --data_path data/hr_data.csv --out_dir results --seed 42
```

Arguments:

- data_path: Path to the input CSV data file.
- out_dir: Directory to save all output results.
- seed: Random seed for reproducibility.

**📈 Outputs**

After running the pipeline, the following results will be generated in the results/ folder:

- Exploratory Analysis:

-- km_overtime.png, km_department.png: Kaplan-Meier survival curves for group comparisons.

- Models & Evaluation:

-- cox_summary.csv: Summary table of coefficients and statistical significance from the CoxPH model.
-- employee_risk_table_3_6_12m.csv: Predicted risk stratification table for each employee at 3, 6, and 12-month horizons.
-- xgb_logloss_overfitting.png: Train/test logloss tracking plot for overfitting detection.
-- shap_summary.png: SHAP summary plot ranking feature impacts on attrition risk.
-- metrics.json: JSON file summarizing key performance metrics (C-index).

**📊 Key Results Summary**

- Cox C-index (train/test): 0.983/0.974
- RSF C-index (train/test): 0.714/0.704
- Notes: AUC(t)/Brier may be unstable under censoring; we rely on C-index as primary.

Top Influential Factors (according to SHAP):

- OverTime_Yes: Overtime significantly increases attrition risk.
- StockOptionLevel: Lower stock option levels are associated with higher risk.
- Age & MonthlyIncome: Younger employees and those with lower income show higher risk.
- YearsAtCompany: Longer tenure is typically associated with lower risk.

**⚠️ Notes & Limitations**

1. Data: The IBM HR Analytics dataset is synthetic/demo data. Patterns and relationships may differ from real-world organizational HR systems.

2. Model Evaluation: Some metrics like time-dependent AUC (AUC(t)) or Brier Score can be unstable at specific time horizons due to censoring. Therefore, we prioritize the C-index as the primary metric for model comparison.

3. Future Work: Potential for more systematic hyperparameter tuning for RSF and XGBoost, experimenting with other models like DeepSurv, or integrating additional features such as performance review history.


