# Fraud Detection

The goal of this project is to analyze bank transactions and determine whether a given payment is fraudulent or not.
The project utilizes Sklearn, Feature Engineering, Feature Selection, and MLflow for model training, while Dagshub is used as a platform for model versioning and logging.

# Fraud Detection Pipeline

The main objective of this project was to build a **fraud detection pipeline**. This pipeline includes several data processing steps, including: data cleaning, feature engineering, and recursive feature selection. Finally, using the processed data, a model is trained and its various metrics are stored as an **MLflow** experiment.

## Project Structure

- `model_experiment_{model_architecture}.ipynb`  
  - Uses a pre-built Pipeline to process training data and train the model

- `model_inference.ipynb`  
  - Contains a ready-made Pipeline that automatically processes test data
  - Previously trained models are evaluated using various metrics, and based on these metrics, we select the best model
  - Which turned out to be XGBoost 
  - **Result: 0.82512**

- `README.md`  
  - Contains a brief overview of the project

## 📁 Dataset

- **Source:** [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)
- The data is split into two parts:
  - `train_transaction.csv`
  - `train_identity.csv`

They are merged using the `TransactionID` column.

---

## 🧪 Pipeline Structure

### 1. `DropMissing`
- **Purpose:** Remove columns that are missing 90% of data.
- **Custom Transformer**

### 2. `FillNaN`
- **Purpose:**
  - Fill NaN values in categorical data with `'missing'`
  - Fill numerical values with the **mode**
- **Custom Transformer**

### 3. `RemoveOutliers`
- **Purpose:** Remove **outliers** from numerical values using Winsorization
- **Custom Transformer**

### 4. `BinaryNonBinaryEncoder`
- **Purpose:**
  - Categorical data is split into two types: Binary (columns with two unique values) and Non-binary
  - Binary data undergoes One-Hot Encoding
  - Non-binary data undergoes Target Encoding
- **Custom Transformer**

### 5. `CorrelationRemover`
- **Purpose:** Remove one feature from pairs of highly correlated columns (correlation threshold of 85%)
- **Custom Transformer**

### 6. `XGBRFE`
- **Purpose:** Recursive Feature Selector that uses XGBoost regressor and selects the **top 80%** of features
- **Custom Transformer**
