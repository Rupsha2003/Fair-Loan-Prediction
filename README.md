# Fair Loan Prediction – A Fairness-Aware Machine Learning Project

## 📌 Project Overview

Loan approval systems play a critical role in financial decision-making, yet they often face criticism for introducing biases against protected groups (e.g., gender, race, age). This project focuses on **Fair Loan Prediction** using machine learning models while applying fairness-aware techniques to ensure equitable outcomes.

The goal is two-fold:

1. **Predict loan approvals accurately.**
2. **Ensure fairness across sensitive attributes** like gender and marital status.

---

## 🎯 Objectives

* Develop predictive models for loan approval classification.
* Investigate dataset biases related to sensitive attributes.
* Apply fairness evaluation metrics and bias mitigation techniques.
* Compare model performance before and after fairness interventions.

---

## 📂 Dataset

The dataset includes applicant details such as:

* Applicant Income
* Loan Amount
* Credit History
* Gender
* Marital Status
* Education Level
* Employment Status

**Target Variable:** Loan Status (Approved/Not Approved)

---

## ⚙️ Methodology

### 🔹 Step 1: Data Preprocessing

* Handling missing values
* Encoding categorical variables
* Normalization of numeric features
* Splitting data into train/test sets

### 🔹 Step 2: Exploratory Data Analysis (EDA)

We examined correlations, class distributions, and sensitive attributes.

Example visualization:

![Loan Distribution by Gender](graph_1.png)

### 🔹 Step 3: Model Training

Implemented models including:

* Logistic Regression
* Random Forest
* XGBoost

Performance was evaluated using **accuracy, precision, recall, and F1-score**.

![Model Accuracy Comparison](graph_2.png)

### 🔹 Step 4: Fairness Analysis

Bias was analyzed using fairness metrics such as:

* **Demographic Parity**
* **Equal Opportunity**
* **Disparate Impact**

![Fairness Metric Results](graph_3.png)

### 🔹 Step 5: Bias Mitigation Techniques

We applied:

* Reweighing sensitive groups
* Threshold adjustments
* Influence-aware reweighting (experimental)

![Bias Mitigation Comparison](graph_4.png)

### 🔹 Step 6: Results & Insights

Post-mitigation, models showed improved fairness scores with a marginal trade-off in accuracy.

![Final Fair vs Non-Fair Models](graph_5.png)

---

## 📊 Key Results

* **Baseline Models**: High accuracy but showed gender bias.
* **Fairness-Aware Models**: Reduced disparity across groups.
* **Trade-off**: Slight reduction in accuracy for significant fairness gain.

---

## 🛠️ Tech Stack

* **Python** (NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn)
* **Fairlearn** for fairness analysis
* **XGBoost** for advanced modeling
* **Jupyter Notebook** for experimentation

---

## 🚀 How to Run

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/Fair_Loan_Prediction.git
   cd Fair_Loan_Prediction
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:

   ```\
   jupyter notebook Fair_Loan_prediction.ipynb
   ```

---

## 🔍 Insights & Future Work

* Fairness-aware modeling is essential for ethical AI systems.
* Future improvements:

  * Apply **counterfactual fairness** techniques.
  * Extend analysis to larger, real-world datasets.
  * Deploy as a **web application** with bias visualization dashboards.

---

## 🙌 Acknowledgments

* **Dataset:** Publicly available loan prediction dataset.
* **Libraries:** Fairlearn, Scikit-learn, XGBoost.
* **Inspiration:** Research on fairness in AI for finance.

---

📢 *This project emphasizes responsible AI development by balancing accuracy with fairness in decision-making systems.*
