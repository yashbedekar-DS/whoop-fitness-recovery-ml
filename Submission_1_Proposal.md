# Submission 1 Proposal

## Project Title

Supervised Learning for WHOOP Fitness Recovery Score Prediction

## Dataset Name, Source Link, Rows, And Columns

| Item | Details |
|---|---|
| Dataset name | WHOOP Fitness Dataset |
| Source link | https://www.kaggle.com/datasets/likithagedipudi/whoop-fitness-dataset |
| Local file | `data/whoop_fitness_dataset_100k.csv` |
| Number of rows | 100,000 |
| Number of columns | 41 |

## Target Variable

`recovery_score`

## Type Of Task

Regression. The target is a continuous recovery/readiness score.

## Problem Statement

Wearable fitness platforms collect daily sleep, strain, workout, heart-rate, and physiological measurements. This project builds supervised regression models to predict WHOOP recovery score and identify which wearable indicators are most useful for recovery planning and fitness decision-making.

## Research Questions

1. How effectively can baseline supervised learning models predict WHOOP recovery score?
2. Which supervised learning model achieves the best predictive performance for recovery score?
3. How do preprocessing strategies affect supervised learning performance?
4. Which features contribute most to recovery score prediction?
5. How does model ranking change across MAE, RMSE, and R2?
6. How robust is the best supervised learning model under cross-validation, numeric noise, and simulated missingness?
7. Which model is most useful, interpretable, reliable, and practical for recovery-score decision support?

## Proposed Methodology

1. Load the WHOOP Fitness dataset.
2. Use `recovery_score` as the target variable.
3. Drop identifier/date leakage fields and engineer date month/day-of-year.
4. Preprocess numerical and categorical features with scikit-learn pipelines.
5. Apply imputation, scaling, and one-hot encoding where required.
6. Train baseline, regularized, tree-based, and ensemble regression models.
7. Evaluate with MAE, RMSE, R2, residual plots, actual-vs-predicted plots, cross-validation, and robustness tests.
8. Save tables and visual outputs for RQ1-RQ7.

## Models

Dummy Regressor, Linear Regression, Ridge Regression, ElasticNet, Decision Tree, k-NN, Random Forest, Gradient Boosting, and XGBoost.

## Evaluation Metrics

MAE, RMSE, R2, residual analysis, actual-vs-predicted plots, cross-validation performance, and robustness under noise/missingness.

## Expected Figures And Tables

Baseline performance tables, model comparison tables, preprocessing comparison tables, feature importance tables, model ranking tables, robustness tables, decision matrix, actual-vs-predicted plots, residual plots, feature importance charts, heatmaps, and robustness charts.

## Consistency With Implemented Code

The proposal matches the notebooks, dataset, target variable, models, evaluation metrics, generated result tables, and visual outputs in this repository.
