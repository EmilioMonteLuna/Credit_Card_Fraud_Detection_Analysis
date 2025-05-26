# Credit Card Fraud Detection

## Overview

Credit card fraud is a major challenge for financial institutions, resulting in billions of dollars in losses annually. This project leverages a real-world, highly imbalanced dataset to build and evaluate machine learning models for the detection of fraudulent transactions. The workflow covers the full data science pipeline, from data exploration and preprocessing to advanced modeling, evaluation, and business recommendations.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Results](#key-results)
- [Business Impact](#business-impact)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Next Steps](#next-steps)
- [Author](#author)
- [Acknowledgements](#acknowledgements)

---

## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Description:** Contains 284,807 anonymized credit card transactions made in September 2013 by European cardholders. Only 492 transactions (0.172%) are labeled as fraud, making this a highly imbalanced dataset.
- **Features:** 30 input features (V1-V28 are PCA components, plus `Time`, `Amount`), and `Class` (target: 1=fraud, 0=not fraud).

---

## Project Workflow

1. **Exploratory Data Analysis (EDA)**
    - Checked for missing values and explored data distributions.
    - Visualized class imbalance and key feature relationships.
    - Identified features most correlated with fraudulent activity.

2. **Data Preprocessing**
    - Scaled `Amount` and `Time` features for improved modeling.
    - Performed stratified train-test split to preserve class proportions.

3. **Handling Class Imbalance**
    - Applied SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic fraud samples in the training data, addressing the extreme class imbalance.

4. **Modeling**
    - **Baseline Model:** Random Forest with class weighting.
    - **Advanced Model:** XGBoost trained on SMOTE-resampled data.
    - Compared both models using precision, recall, F1-score, ROC-AUC, and precision-recall curves.

5. **Feature Importance**
    - Analyzed and visualized the most important features for fraud detection in both models.

6. **Threshold Tuning & Business Analysis**
    - Explored the impact of adjusting the classification threshold on recall and precision.
    - Performed a cost-benefit analysis to estimate the financial impact of false negatives (missed frauds) and false positives (false alarms).

7. **Conclusions & Recommendations**
    - Summarized findings and provided actionable business recommendations for deploying fraud detection models in a real-world setting.

---

## Key Results

- **Class Imbalance:** Only 0.17% of transactions are fraudulent.
- **Model Performance:**
    - **Random Forest:** Provided a strong baseline, but struggled with recall for the minority class.
    - **XGBoost + SMOTE:** Achieved ROC-AUC of 0.98 and strong recall/precision balance, outperforming the baseline.
- **Feature Importance:** Features V17, V14, V12, and V10 were identified as the most predictive of fraud.
- **Business Recommendation:** Threshold tuning and cost analysis allow financial institutions to balance fraud loss prevention with customer experience.

---

## Business Impact

Deploying a robust fraud detection model can:
- Significantly reduce financial losses due to fraudulent transactions.
- Improve customer trust and satisfaction by minimizing false alarms.
- Enable data-driven decision-making for risk management teams.
- Provide actionable insights into the patterns and features most indicative of fraud, informing further investigation and prevention strategies.

---

## How to Run

1. **Clone this repository:**
    ```
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```
2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
3. **Download the dataset:**
    - Get the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project folder.

4. **Run the notebook:**
    - Open `credit-card-fraud-detection.ipynb` in Jupyter Notebook or JupyterLab and execute the cells sequentially.

---

## Technologies Used

- **Python 3**
- **pandas, numpy** (data manipulation)
- **matplotlib, seaborn** (visualization)
- **scikit-learn** (modeling, preprocessing, metrics)
- **imbalanced-learn** (SMOTE for class imbalance)
- **XGBoost** (advanced modeling)

---

## Next Steps

- Hyperparameter tuning with Optuna or GridSearchCV for further model optimization.
- Explore additional resampling techniques (e.g., ADASYN, Tomek Links).
- Investigate anomaly detection methods (e.g., Isolation Forest, Autoencoders).
- Develop a real-time fraud detection API or dashboard for deployment.
- Collaborate with domain experts to interpret key features and refine business rules.

---
## Acknowledgements

- [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- scikit-learn, XGBoost, imbalanced-learn open source communities

---
