# Employee Attrition Prediction

Predicting which employees are likely to leave (Attrition) and explaining *why* using SHAP, based on the IBM HR Employee Attrition dataset.

## Overview

The notebook (`ML_Project_Final.ipynb`) walks through a full classification workflow: EDA, cleaning, preprocessing, model training with proper SMOTE handling inside cross-validation, hyperparameter tuning, model comparison, and explainability with SHAP.

**Dataset:** `IBM_Employee_Attrition_Dataset.csv` — 1,470 employee records, 35 columns (34 features + `Attrition` target). The target is imbalanced (~84% "No" / ~16% "Yes").

## Workflow

### 1. Exploratory Data Analysis
- Class distribution, nulls, duplicates, and unique-value checks
- Numerical feature distributions (histograms, KDE plots by attrition status)
- Correlation of numeric features with the target
- Categorical feature attrition rates (`OverTime`, `BusinessTravel`, `MaritalStatus`, `EducationField`, `JobRole`, etc.)
- Satisfaction/involvement ordinal features vs. attrition trend
- Full correlation heatmap

**Key insight:** `OverTime = Yes` is one of the strongest predictors, along with frequent travel, single marital status, and certain job roles.

### 2. Data Cleaning
- Dropped non-informative columns (`EmployeeNumber`, `Over18`, `EmployeeCount`, `StandardHours`)
- IQR-based outlier treatment on skewed numeric columns (income, tenure-related fields, etc.)

### 3. Preprocessing
- One-hot encoding for categorical features (`ColumnTransformer` + `OneHotEncoder`)
- Label encoding for the target
- 80/20 stratified train/test split

### 4. Modeling
Each model is trained two ways: a baseline pass, then hyperparameter tuning via `GridSearchCV` (5-fold CV, scoring = **recall**, chosen as the primary business metric since missing an at-risk employee is costlier than a false alarm). SMOTE oversampling is applied *inside* the CV pipeline (`imblearn.pipeline.Pipeline`) to prevent leakage from resampling before the train/validation split.

| # | Model | Notes |
|---|-------|-------|
| 1 | Logistic Regression | Baseline + tuned (L1/L2, C grid) |
| 2 | Random Forest | Baseline + tuned (n_estimators, depth, leaf/split sizes, max_features) |
| 3 | XGBoost | Full-feature tuned model, then a reduced model retrained on the top-10 features by importance |
| 4 | Decision Tree | Baseline + tuned (depth, leaf/split sizes, criterion) |
| 5 | KNN | Baseline + tuned (k, weights, distance metric) |
| 6 | LightGBM | Tuned (n_estimators, learning rate, depth, num_leaves) |
| 7 | MLP (Neural Network) | Tuned (hidden layer sizes, activation, alpha) |

Each model reports training/test accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

### 5. Model Comparison
Bar chart comparisons across all tuned models:
- Train vs. test accuracy (overfitting check)
- Test F1 / Recall / ROC-AUC
- Test F1 / Recall / Precision

### 6. Explainability (SHAP)
- SHAP feature importance for the best Logistic Regression model
- SHAP feature importance for the tuned MLP model
- SHAP summary plot

### 7. Loss Curves
Training loss curves for iterative learners (MLP, XGBoost) to visually assess convergence.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
```

## Usage

1. Place `IBM_Employee_Attrition_Dataset.csv` in the same directory as the notebook.
2. Install dependencies (see above).
3. Run all cells top to bottom — later sections (model comparison, SHAP) depend on variables produced earlier (best-tuned model objects and metric variables per model).

## Notes

- Recall is the optimization target throughout, reflecting a business preference for catching at-risk employees over minimizing false positives.
- SMOTE is applied only within the CV pipeline / training fold, never on the held-out test set, to avoid data leakage.
- The XGBoost section includes an extra step: refitting on the top-10 most important features to see whether a reduced feature set retains performance.
