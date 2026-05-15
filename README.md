# Submission 1: Proposal, Code, and GitHub Repository

## Project Title

Supervised Learning for WHOOP Fitness Recovery Score Prediction

## Dataset Link

Original public dataset: [WHOOP Fitness Dataset](https://www.kaggle.com/datasets/likithagedipudi/whoop-fitness-dataset)

Local dataset file:

```text
data/whoop_fitness_dataset_100k.csv
```

## Project Overview

This project uses supervised machine learning to predict `recovery_score` from WHOOP-style fitness, sleep, heart-rate, strain, workout, and user profile indicators.

## Dataset Details

| Item | Value |
|---|---|
| Dataset name | WHOOP Fitness Dataset |
| Source | Kaggle |
| Rows | 100,000 |
| Columns | 41 |
| Target variable | `recovery_score` |
| Task type | Regression |
| Application domain | Fitness, recovery, and wearable health analytics |

## Research Questions

1. How effectively can baseline supervised learning models predict WHOOP recovery score?
2. Which supervised learning model achieves the best predictive performance for recovery score?
3. How do preprocessing strategies affect model performance?
4. Which input features contribute most to recovery score prediction?
5. How does model ranking change across MAE, RMSE, and R2?
6. How robust is the best model under cross-validation, noise, and missing-data conditions?
7. Which model provides the best practical balance of accuracy, interpretability, reliability, runtime, and deployment suitability?

## How To Run The Code

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run all outputs:

```bash
python whoop_assignment_pipeline.py --run-all
```

Or open and run the notebooks in `notebooks/` from `RQ1_notebook.ipynb` to `RQ7_notebook.ipynb`.

## Generated Outputs

Tables are saved in `outputs/tables/`. Figures are saved in `outputs/figures/images/`.

## Folder Structure

The main project lives in `Machine_Learning_Assignment/` with a cleaner layout:

- `data/` for the dataset
- `notebooks/` for the RQ1-RQ7 notebooks
- `outputs/tables/` for generated CSV tables
- `outputs/figures/images/` for generated PNG figures
- `reports/overleaf/` for the Overleaf-ready LaTeX submission package

## Key Final Result

The final recommended model is `Linear Regression` because it provides the best weighted balance across accuracy, reliability, interpretability, runtime, and deployment suitability.

## Submission Checklist

- Proposal file: `Submission_1_Proposal.docx`
- Markdown proposal: `Submission_1_Proposal.md`
- Complete code: `whoop_assignment_pipeline.py` and `notebooks/RQ1_notebook.ipynb` to `notebooks/RQ7_notebook.ipynb`
- README file: `README.md`
- Dataset link: https://www.kaggle.com/datasets/likithagedipudi/whoop-fitness-dataset
- Generated outputs: `outputs/`
- Required libraries: `requirements.txt`
