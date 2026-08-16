# Medical Insurance Cost Prediction

Predicting how much someone's medical insurance charges will be, based on a handful of everyday details — age, BMI, whether they smoke, how many kids they have, and so on.

## What's in here

- `Medical_Insurance_Cost_Prediction.ipynb` — the full notebook, from raw data to a tuned, compared set of models
- `insurance.csv` — the dataset (1,338 rows, 7 columns)

## What the notebook actually does

1. **Explores the data** — quick look at distributions, missing values, and how features like age, BMI, and smoking status relate to charges
2. **Encodes categorical columns** (sex, smoker, region) so models can use them
3. **Trains 5 models**: Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting
4. **Scales features properly** for Ridge/Lasso using a `Pipeline`, so cross-validation doesn't leak information between folds
5. **Tunes hyperparameters** for every model using `GridSearchCV` with 5-fold cross-validation
6. **Compares everything** — RMSE, MAE, and R² for both training and test data, plus a bar chart to make it visual
7. **Builds a simple predictive system** at the end, so you can plug in a person's details and get a charge estimate

## Results (test data)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Gradient Boosting (Tuned) | 0.870 | ~4,417 | ~2,419 |
| Random Forest (Tuned) | 0.870 | ~4,419 | ~2,351 |
| Gradient Boosting | 0.868 | ~4,445 | ~2,376 |
| Random Forest | 0.837 | ~4,948 | ~2,795 |
| Linear / Ridge / Lasso | ~0.745 | ~6,190 | ~4,270 |

**Takeaway:** tree-based models (Random Forest, Gradient Boosting) clearly win here. Charges don't move in a straight line with age or BMI — a smoker with a high BMI sees a much bigger jump in cost than a straight-line model can capture, and tree-based models pick up on that kind of pattern much better.

## Running it yourself

Keep `insurance.csv` in the same folder as the notebook, then just run all cells top to bottom. Needs `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` — nothing exotic.

## Notes

- Scaling is only applied for Ridge and Lasso, since Linear Regression's R² isn't affected by it and tree-based models don't need it at all.
- Hyperparameter tuning gave Random Forest and Gradient Boosting a real boost, but barely moved the needle for Ridge/Lasso — they didn't have much overfitting to fix in the first place, given there are only 6 features here.
