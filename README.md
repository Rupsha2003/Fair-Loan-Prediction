# Fair Loan Prediction – A Fairness & Explainability-focused ML Project

## 📌 Project Overview

This repository implements a loan approval prediction pipeline with a strong emphasis on **explainability** and **fairness-aware evaluation**. The notebook focuses on two primary models:

* **Support Vector Machine (SVM)** (used as a strong baseline classifier)
* **Explainable Boosting Machine (EBM)** implemented via `interpret`'s `ExplainableBoostingClassifier` (used as the main explainable model)

Additional baseline models (Random Forest, XGBoost) are included for comparison, but the notebook centers on SVM and EBM.

---

## 🎯 Goals

* Build accurate classifiers for loan approval prediction.
* Prioritize model interpretability using EBM and SHAP.
* Evaluate fairness across sensitive attributes and document model behavior.
* Save tuned models and supporting artifacts for reproducibility.

---

## 📂 Dataset (summary)

Typical columns used in the notebook include (may vary depending on your raw CSV):

* `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`
* Demographic / categorical variables: `Gender`, `MaritalStatus`, `Employment_Status`, `Education`, etc.

**Target:** `Loan_Status` (binary: Approved / Not Approved)

---

## 🧭 Notebook Structure

* `0_data_loading_and_eda` — data import, basic cleaning, EDA (distributions, missingness)
* `1_preprocessing` — encoding (`pd.get_dummies` / `OneHotEncoder` for categorical features), handling missing values
* `2_feature_prep` — train/test split (`train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`)
* `3_baseline_models` — Random Forest, SVM (`SVC(kernel='rbf', probability=True)`), XGBoost for comparison
* `4_ebm_training_and_tuning` — `ExplainableBoostingClassifier` + `GridSearchCV` for hyperparameter tuning
* `5_explainability` — SHAP analysis and EBM native explanations
* `6_evaluation_and_artifacts` — metrics (accuracy, ROC-AUC, confusion matrix), saving models and images

---

## ⚙️ Preprocessing & Experimental Details

* **Encoding:** Categorical variables are encoded with `pd.get_dummies()` (e.g., `Employment_Status`) and/or `OneHotEncoder` where appropriate.
* **Train/Test Split:** 80/20 split performed with `stratify=y` to preserve class ratios.
* **Scaling:** SVM benefits from feature scaling — check notebook cells for `StandardScaler` if used; otherwise apply scaling before SVM training.
* **SHAP compatibility:** The notebook converts boolean columns to integers when running SHAP explanations.

---

## 🔬 Models & Tuning

### Support Vector Machine (SVM)

* Implemented via `sklearn.svm.SVC(kernel='rbf', probability=True, random_state=42)`.
* Used as a baseline; predictions and classification report are printed in the notebook.
* If you plan to productionize SVM, scale the features and consider a `GridSearchCV` on `C` and `gamma`.

### Explainable Boosting Machine (EBM)

* Implemented using `interpret.glassbox.ExplainableBoostingClassifier`.
* Notebook runs a `GridSearchCV` over parameters such as `interactions`, `learning_rate`, `max_bins`, and `min_samples_leaf` to find a tuned EBM.
* EBM is saved as `best_ebm_model.joblib` / `best_ebm_model.pkl` in the notebook.
* EBM provides global and per-feature explanations; the notebook also computes SHAP values for further insight.

### Other Baselines

* **Random Forest** (`sklearn.ensemble.RandomForestClassifier`) and **XGBoost** (`xgboost.XGBClassifier`) are trained as comparison models. Use them to confirm if a non-interpretable model significantly outperforms EBM.

---

## 📈 Explainability & Fairness

* **EBM native explainability**: partial dependence-like plots and per-feature contributions are used to understand model reasoning.
* **SHAP analysis**: a SHAP `Explainer` is built (using `model.predict_proba`) to produce feature importance and per-instance explanations. The notebook includes code to handle boolean/int dtype issues for SHAP.
* **Fairness checks**: the notebook includes exploratory checks on sensitive attributes (for example `Gender`) — use the EDA cells to inspect disparities. If you want automated fairness metrics (e.g., demographic parity, equal opportunity), you can integrate `fairlearn` or compute group-level rates manually.

---

## 🔧 Artifacts Saved by the Notebook

Typical files created by the notebook (check the final cells):

* `best_ebm_model.joblib` / `best_ebm_model.pkl` — persisted EBM model
* `X_train_sample.csv`, `y_train_sample.csv` — small snapshots of training data
* `roc_curve_ebm_tuned.png`, `confusion_matrix_ebm_tuned.png` — evaluation visuals for the tuned EBM
* `model_info.json` — optional metadata summary (if generated)

---

## 📊 Embedded Visuals (exported from the notebook)

> The notebook output images have been embedded below so readers can inspect model behavior and results quickly.


<img width="633" height="687" alt="image" src="https://github.com/user-attachments/assets/610ec61c-5eaa-4350-a3ea-1b396e28306a" />
*Figure: Model performance comparison (SVM / EBM / RF / XGBoost).*

<img width="852" height="410" alt="image" src="https://github.com/user-attachments/assets/0ffb34d3-e11b-4201-add3-d515109006b7" />
*Figure: Example fairness-related metrics or group-level comparisons.*

<img width="667" height="471" alt="image" src="https://github.com/user-attachments/assets/a7756234-ccaa-4830-a854-8cc1c8ea5e1d" />

*Figure: Tuned EBM ROC curve and confusion matrix visuals.*

<img width="633" height="687" alt="image" src="https://github.com/user-attachments/assets/b38d8bc7-d418-4da4-b42d-90ae70c8cdf4" />

*Figure: SHAP/feature importance summary (example).*

> **Note:** Filenames here match the exported notebook outputs. If you move images into an `images/` folder in your repo, update the image paths accordingly.

---

## 🛠️ How to Reproduce / Run Locally

1. Clone the repo and change directory:

```bash
git clone https://github.com/yourusername/Fair_Loan_Prediction.git
cd Fair_Loan_Prediction
```

2. Install dependencies (example `requirements.txt` should include `scikit-learn`, `interpret`, `shap`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`):

```bash
pip install -r requirements.txt
```

3. Start Jupyter and run the notebook:

```bash
jupyter notebook "Fair_Loan_prediction (2).ipynb"
```

4. Re-run cells in order. Check the final cells for saved artifacts (models, plots).

---

## 🔎 Suggested Next Steps / Improvements

* **Explicit fairness metrics:** Integrate `fairlearn` to compute demographic parity, disparate impact, and equal opportunity automatically.
* **SVM tuning:** Run `GridSearchCV` over `C`, `gamma`, and `kernel` for SVM to optimize performance.
* **Calibration & thresholds:** For production, consider probability calibration (e.g., `CalibratedClassifierCV`) and threshold-tuning per group if fairness constraints are required.
* **Deployment:** Wrap the best model (EBM or tuned SVM) in a small FastAPI app and add a dashboard for per-request explanations.

---






