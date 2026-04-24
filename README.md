# Predicting-Student-Academic-Outcomes

This project trains machine learning models to predict student academic outcomes from `data.csv`.

## Run the model script

From the project root, run:

```bash
sklearn-env/bin/python project.py
```

This will:
- load and preprocess the dataset
- train and compare the models
- save evaluation results to CSV files
- save Random Forest feature importance outputs

## Main output files

- `model_comparison_results.csv`
- `best_model_confusion_matrix.csv`
- `random_forest_tuning_comparison.csv`
- `random_forest_feature_importance.csv`
- `random_forest_feature_importance_top10.png`
