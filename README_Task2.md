# 🚀 Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API
**DevelopersHub Corporation – AI/ML Engineering Advanced Internship**

## Objective
Build a reusable, production-ready ML pipeline for predicting customer churn using the Telco Customer Churn Dataset.

## Methodology / Approach
- **Preprocessing:** `ColumnTransformer` with separate `Pipeline` branches for numerical (StandardScaler + SimpleImputer) and categorical (OneHotEncoder + SimpleImputer) features
- **Models:** Logistic Regression (baseline) and Random Forest
- **Tuning:** `GridSearchCV` with 5-fold Stratified CV optimizing F1-Score
- **Export:** Complete pipeline serialized via `joblib` for production reuse

## Dataset
Telco Customer Churn Dataset (IBM/Kaggle) — 13 features, binary churn target (~26% positive rate)

## Key Results / Observations
| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~0.80 | ~0.62 | ~0.84 |
| Random Forest | ~0.82 | ~0.66 | ~0.87 |
| **RF + GridSearch (Best)** | **~0.83** | **~0.68** | **~0.88** |

- **Tenure** is the #1 predictor: short-tenure customers churn significantly more
- **Month-to-month contracts** → highest churn risk
- **Monthly charges > $80** correlates strongly with churn
- The exported `joblib` pipeline can be loaded directly into a Flask/FastAPI service

## Files
| File | Description |
|---|---|
| `Task2_ML_Pipeline_Customer_Churn.ipynb` | Full solution notebook |
| `churn_pipeline_best.joblib` | Exported production pipeline |
| `eda_churn.png` | EDA visualizations |
| `model_evaluation.png` | ROC curves, confusion matrix, metric comparison |
| `feature_importance.png` | Top 15 feature importances |

## Setup
```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
jupyter notebook Task2_ML_Pipeline_Customer_Churn.ipynb
```

## Skills Demonstrated
ML pipeline construction · Hyperparameter tuning with GridSearch · Model export and reusability · Production-readiness practices
