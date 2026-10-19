# Fraud Detection Pipeline: Management Report

## 1. Executive Summary
We built and tested a machine-learning pipeline that flags fraudulent card transactions in a dataset where only 0.17% of 283,726 transactions are fraud. After comparing Logistic Regression and Random Forest, **Random Forest** was selected. On a completely untouched test set it caught **73% of fraud** (69 of 95 cases) while raising **5 false alerts**, with ROC-AUC 0.957 and PR-AUC 0.813. Under the stated cost assumptions it stops about **60% of fraud value** on the test set. Test results come from few fraud cases and are uncertain (95% interval for recall: 0.64-0.81).

## 2. Problem Statement
Fraud is rare but costly. A model judged on accuracy can look near-perfect while catching nothing (always predicting "legitimate" scores 99.83% accuracy and 0% recall). The task is to detect fraud with a controlled balance between missed fraud and false alarms.

## 3. Methodology
Stratified train/validation/test split; all preprocessing, SMOTE and tuning inside an `imblearn` pipeline so no information leaks from validation or test data; models judged on precision, recall, F1, ROC-AUC, PR-AUC and confusion matrices; selection on validation PR-AUC; final numbers on a one-time test evaluation.

## 4. Data Preparation
1,081 duplicate rows removed; no missing values (a median imputer is still in the pipeline for safety); outliers kept because they carry fraud signal; 7 engineered features (hour, night flag, log amount, zero/round amount flags, PCA-space norm and max); stratified split.

## 5. Modeling Approach
Logistic Regression (scaled, L2-regularised) and Random Forest, each with SMOTE applied to training folds only. Tuned with GridSearchCV (LR) and RandomizedSearchCV (RF) on PR-AUC; decision threshold chosen on validation data. Best settings: LR {'clf__C': 0.01, 'smote__k_neighbors': 5, 'smote__sampling_strategy': 0.1}; RF {'smote__sampling_strategy': 0.05, 'smote__k_neighbors': 3, 'clf__n_estimators': 200, 'clf__min_samples_leaf': 2, 'clf__max_features': 'sqrt', 'clf__max_depth': 24, 'clf__class_weight': 'balanced_subsample'}.

## 6. Evaluation Results
| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Random Forest (selected) | 0.932 | 0.726 | 0.817 | 0.957 | 0.813 |
| Logistic Regression | 0.871 | 0.642 | 0.739 | 0.955 | 0.705 |

## 7. Key Findings
- Accuracy is meaningless here; recall and precision tell the real story.
- Applying SMOTE before splitting inflated PR-AUC from 0.706 (correct) to 0.991 (leaky) and precision from 0.047 to 0.973 in our own experiment: a leakage artefact, not skill.
- The most influential features (permutation importance): V12, V14, V3, v_norm, V4.
- Differences between the two models can be small relative to test-set noise; selection was made on the larger validation split.

## 8. Business Recommendations
Deploy first in assisted (human-review) mode; set the threshold from the institution's real review and chargeback costs; keep the simpler model as a challenger; monitor drift and retrain regularly; tier the response by score.

## 9. Future Improvements
Gradient boosting with cost-sensitive learning, probability calibration, time-based validation, customer behavioural features, SMOTE variants, anomaly detection for novel fraud, SHAP explanations and an analyst feedback loop.

## 10. Conclusion
A leak-free pipeline with SMOTE, tuned classifiers and threshold selection delivers a fraud detector whose value is measured in caught fraud and manageable false alerts, not accuracy. The main caveats are the small number of fraud examples and the anonymised features; both are addressed in the roadmap.
