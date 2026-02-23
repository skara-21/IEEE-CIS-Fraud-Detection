# IEEE-CIS Fraud Detection Pipeline

A machine learning pipeline for detecting fraudulent bank transactions, built on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) from Kaggle. The project achieves an **AUC-ROC of 0.825** on the Kaggle test set using XGBoost, with full experiment tracking via MLflow and DagsHub.


## Overview

Online payment fraud costs businesses billions annually. This project builds an end-to-end ML pipeline — from raw, messy transactional data to a trained, versioned model — with a focus on production-ready design: all preprocessing steps are encapsulated as custom sklearn-compatible transformers, making the pipeline fully reproducible and easy to deploy.

Three models were trained and compared (Logistic Regression, Random Forest, XGBoost). XGBoost was selected as the best performer based on AUC-ROC, which is the appropriate metric for this heavily imbalanced dataset (only ~3.5% of transactions are fraudulent).


## Results

| Model | AUC-ROC |
|---|---|
| Logistic Regression | **0.8986** |
| Random Forest | **0.8539** |
| **XGBoost** | **0.9217** |


Each experiment notebook loads the training data, runs it through the full preprocessing pipeline, trains the model, and logs all metrics and artifacts to MLflow via DagsHub.


## Pipeline Architecture

All preprocessing steps are implemented as **custom sklearn transformers** (inheriting from `BaseEstimator` and `TransformerMixin`), making them fully compatible with sklearn `Pipeline` and enabling clean train/test separation with no data leakage.

### 1. `DropMissing`
Drops columns where more than 90% of values are missing. Threshold is learned at fit time and applied consistently to test data.

### 2. `FillNaN`
Imputes missing values — categorical columns are filled with `'missing'` as an explicit category, and numerical columns are filled with the training set mode. Imputation values are stored at fit time to prevent leakage.

### 3. `RemoveOutliers`
Applies **Winsorization** to numerical columns, capping extreme values at the 1st and 99th percentiles. This is preferred over dropping outliers as it preserves all transactions while reducing the influence of anomalous values.

### 4. `BinaryNonBinaryEncoder`
Handles categorical encoding intelligently by splitting columns into two groups:
- **Binary columns** (≤2 unique values) → One-Hot Encoding
- **Non-binary columns** (>2 unique values) → WOE (Weight of Evidence) Encoding, which is standard practice in credit risk and fraud detection as it captures the relationship between each category and the fraud rate directly

### 5. `CorrelationRemover`
Computes the pairwise correlation matrix and removes one feature from any pair with correlation above 85%, reducing multicollinearity before feature selection.

### 6. `XGBRFE`
Recursive Feature Elimination using an XGBoost regressor as the estimator. Selects the top 80% of features by importance, significantly reducing dimensionality while retaining the most predictive signals.


## Experiment Tracking

All training runs are tracked with **MLflow**, hosted on **DagsHub**. For each experiment, the following are logged:
- Model parameters and hyperparameters
- Evaluation metrics (AUC-ROC, Average Precision, F1, Precision, Recall)
- Trained pipeline artifact (for inference)

This setup allows direct comparison across model architectures and full reproducibility of any run.


## Dataset

**Source:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

The data consists of two files joined on `TransactionID`:
- `train_transaction.csv` — 590,000 transactions with payment and card features
- `train_identity.csv` — device and identity information for a subset of transactions

Key challenges: ~3.5% fraud rate (severe class imbalance), ~414 features with missing values, and 400+ anonymized V-columns with non-normal distributions.


## Tech Stack

- **ML & Preprocessing:** scikit-learn, XGBoost, imbalanced-learn, category_encoders, scipy
- **Data:** pandas, NumPy
- **Experiment Tracking:** MLflow, DagsHub
- **Environment:** Kaggle Notebooks (GPU)


## Possible improvements

- Add SHAP value analysis for model explainability
- Deploy best model as a REST API (FastAPI + Docker)
- Add precision-recall curve analysis
