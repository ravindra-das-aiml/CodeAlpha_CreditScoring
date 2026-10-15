# CodeAlpha_CreditScoring

Credit scoring model that predicts whether a loan applicant is a good or bad credit risk (CodeAlpha ML Internship, Task 1).

## Dataset
Statlog German Credit Data (UCI), 1000 applicants, 20 features (checking account status, loan duration, credit amount, credit history, savings, employment, age, etc.).
Target: 1 = bad credit (high risk), 0 = good credit. Class balance is about 30% bad / 70% good.

## Approach
1. EDA (class balance, credit amount and duration by risk, correlations)
2. Feature engineering from financial history: monthly payment, payment burden (debt-to-income proxy), amount per age, long-loan flag, log amount, age group
3. Preprocessing: scaling and one-hot encoding inside a sklearn Pipeline (no data leakage)
4. Models: Logistic Regression, Decision Tree, Random Forest (class_weight="balanced" for the imbalance)
5. Evaluation: Precision, Recall, F1-Score, ROC-AUC, ROC curves, confusion matrices
6. Random Forest tuning (GridSearchCV), decision-threshold selection on cross-validated training predictions, feature importance

## Results
Metrics are for the bad-credit class (positive class) on the test set (200 applicants, 60 bad).

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.766 | 0.755 | 0.565 | 0.800 | 0.662 | 0.810 |
| Random Forest | 0.795 | 0.745 | 0.562 | 0.683 | 0.617 | 0.800 |
| Decision Tree | 0.697 | 0.605 | 0.402 | 0.650 | 0.497 | 0.635 |

## Tuned Random Forest
Best params: max_depth=10, min_samples_leaf=1, n_estimators=200 (CV ROC-AUC 0.798)
Test: Accuracy 0.78, ROC-AUC 0.81. Bad credit: precision 0.61, recall 0.72, F1 0.66. Good credit: precision 0.87, recall 0.81, F1 0.84.

## Decision threshold
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 (default) | 0.614 | 0.717 | 0.662 |
| 0.43 (tuned) | 0.474 | 0.750 | 0.581 |

Lowering the threshold raised recall slightly but hurt precision and F1, so it did not clearly help.

## Key findings
- Logistic Regression had the best recall (0.80), F1 and test ROC-AUC; Random Forest had the best cross-validated ROC-AUC.
- The Decision Tree was clearly the weakest model.
- Top features: checking account status, monthly payment, log credit amount, amount per age. Four of the top seven features were engineered.

## Limitations
Small dataset (1000 applicants, 200 in the test set), so results vary with the split.
The dataset has no income column, so "payment burden" is only a proxy for debt-to-income.
This is a learning project and not a real lending decision tool.

## Run
```
pip install -r requirements.txt
jupyter notebook credit_scoring.ipynb
```