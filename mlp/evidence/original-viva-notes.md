# MLP project walkthrough

Use the executed notebook and its outputs while preparing. These notes explain
the decisions; the numerical results must be taken from the saved run.

## Trace one row through the notebook

1. Load a labelled listing, preserve `space_id` for output alignment and exclude it from predictors.
2. Hash raw predictor profiles without the ID or target. Keep matching profiles in one partition.
3. Fit caps and the facilities vocabulary on fitting rows, then transform tuning and audit rows with those frozen values.
4. For the linear and Extra Trees baselines, impute numerical missing values and one-hot encode nominal categories. Scale for the linear model.
5. For LightGBM, retain categorical columns with a fitting-only category vocabulary. CatBoost receives strings plus explicit numeric missing sentinels.
6. Use tuning macro F1 for settings and early stopping. Freeze the decision before inspecting audit results.
7. Refit preprocessing and the chosen model on all labels, using the frozen number of boosting rounds. Preserve sample submission IDs exactly.

## Five components of the model families

| Family | Data representation | Model | Loss or fitting criterion | Optimization | Evaluation |
|---|---|---|---|---|---|
| Linear classifier | Scaled numeric and one-hot inputs | Linear class scores | Log loss with L2 regularization | Stochastic gradient descent | Macro F1 and accuracy |
| Extra Trees | Imputed numeric and one-hot inputs | Randomized tree ensemble | Impurity reduction at splits | Recursive random candidate splits and averaging | Same split and metrics |
| LightGBM | Numeric and categorical engineered inputs | Additive boosted trees | Multiclass log loss | Gradient boosting with histogram splits | Tuning macro F1 for early stopping |
| CatBoost | Numeric inputs and native categorical strings | Additive boosted trees | Multiclass log loss | Boosting with categorical processing | Tuning macro F1 and frozen audit |

Macro F1 averages four class F1 scores equally. Accuracy gives each row equal
weight. Explain why these answer different questions, especially with class
imbalance. A validation score is not a public leaderboard score.

## Explain these limitations honestly

The final profile split prevents exact-profile duplicates crossing partitions.
It does not establish generalization to unseen operators or cities. Rare
categories may occur only outside fitting data. Repeated tuning can overfit a
holdout, which is why the audit does not select the winner. Finite-iteration
feature experiments in Milestone 3 produced some convergence warnings; those
are recorded in the report and are not hidden.

The 80,000/20,000 milestone feature study and the final full-data model study
serve different purposes. Do not compare their scores as if they were measured
on the same validation rows or claim that a linear-model feature gain must
transfer to a nonlinear model.

## Practice without looking at the implementation

- Implement macro F1 from a four-class confusion matrix.
- Fit a group mean on training predictors and apply it to an unseen validation group with a fallback.
- Explain why fitting an imputer or vocabulary before splitting leaks information.
- Derive the effect of an L2 penalty on a linear classifier's gradient.
- Explain how boosting differs from independently fitted randomized trees.
- Trace how a test row with a new categorical value reaches the final prediction.
- Show that submission IDs match the sample order and that all predictions are valid tiers.
