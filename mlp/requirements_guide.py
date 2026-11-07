"""Map the learner's declarations to exact code and recorded outputs."""
import html
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

CHECKS = [
 dict(id='three-models', title='Used at least three different models', mandatory=True,
 meaning='A model is a learned mapping from listing attributes to a price-tier prediction. Different parameter settings of LightGBM are configurations of the same family; we also need genuinely different estimators.',
 how='The final notebook fits SGDClassifier, ExtraTreesClassifier, LightGBM and CatBoost. The shared record function calls both fit and predict for every configuration. Five recorded configurations therefore cover four model families.',
 read='model.fit learns from fitting rows and labels yf. model.predict(tune_data) produces the tuning predictions p. f1_score compares p with the known tuning labels yt. The named record calls tell you which estimator is being tested.',
 result='The saved model table has five rows: linear_log_loss, extra_trees, lightgbm_63, lightgbm_31_regularized and catboost. All have measured tuning and audit scores.',
 snippets=[(10,7,25),(10,28,54),(10,73,102),(10,112,134)], outputs=[(10,2),(12,1)], where='models',
 say='I produced predictions from four model families using the same fit/tune split. LightGBM was tested at two capacities, which are two settings of one family.'),
 dict(id='eda', title='Performed exploratory data analysis (EDA)', mandatory=True,
 meaning='EDA means inspecting the dataset before drawing conclusions from a fitted model: what values exist, what is missing, how labels are distributed and how columns relate.',
 how='Cell 4 separates X and y, counts repeated IDs and profiles, computes missing fractions, plots label/capacity distributions, shows categorical crosstabs and plots numeric correlations.',
 read='isna().mean() counts missing entries as 1 and averages them into a fraction. value_counts() counts labels. The boxplot compares capacity across tiers. These are data descriptions, not model predictions.',
 result='The recorded audit found 0 duplicate identifiers and 1,160 repeated predictor profiles. borough is approximately 76.66% missing. The EDA plot, crosstabs and correlation heatmap are saved outputs of this cell.',
 snippets=[(4,1,20),(4,21,30)], outputs=[(4,1),(4,2),(4,3),(4,7)], where='eda',
 say='I inspected duplicates, missingness, label balance, capacity, categorical tier proportions and numeric correlations. These findings motivated grouped splitting and careful preprocessing.'),
 dict(id='analysis',title='Analyzed model results and performance',
 meaning='Analysis means explaining the scores and mistakes, rather than only displaying a winning number. Precision, recall, F1 and a confusion matrix answer different questions.',
 how='Cell 12 reports audit metrics for each recorded model, prints a class-level report, plots the selected model’s confusion matrix, examines errors by category and estimates permutation importance.',
 read='The audit prediction loop has no fit call. classification_report summarizes predictions against labels. normalize="true" divides each confusion-matrix row by its true-class total, so the diagonal is recall.',
 result='The selected model’s audit macro F1 is 0.738081 and accuracy is 0.741498. Tier 3 is weakest, with F1 about 0.5920 and recall about 0.5684. The guide explains this using the exact saved confusion counts.',
 snippets=[(12,1,31)],outputs=[(12,1),(12,2),(12,3)],where='audit',
 say='I evaluated the frozen choice on audit and looked at per-class and subgroup errors. Tier 3 was the main measured weakness; I did not treat a high overall score as proof that every class worked equally well.'),
 dict(id='tuning',title='Performed hyperparameter tuning',
 meaning='Hyperparameters control learning before fitting, such as tree capacity, minimum leaf support and the number of boosting rounds. Tuning compares settings using development data.',
 how='Cell 10 compares LightGBM with 63 leaves/minimum child support 50 against 31 leaves/support 100. Early stopping uses tuning macro F1 with patience 100. The final winner is chosen by tuning F1.',
 read='The loop changes leaves and child. eval_set contains Lt and yt, the tuning partition. early_stopping(100) monitors that partition. Sorting tune_macro_f1 establishes the winner before the audit cell.',
 result='The 63-leaf setting reached tuning F1 0.735466 with best iteration 1,508. The 31-leaf setting reached 0.728382 with best iteration 1,310. This is a limited, motivated comparison, not an exhaustive grid search or a final CV average.',
 snippets=[(10,73,102),(10,135,138)],outputs=[(10,1),(10,2),(10,3)],where='models',
 say='I tested two LightGBM capacity settings and selected boosting rounds by tuning-based early stopping. The larger setting won on tuning macro F1, so I froze it before audit evaluation.'),
 dict(id='structure',title='Kept the notebook structured and commented',
 meaning='Readable structure separates loading, EDA, splitting, preprocessing, model fitting, evaluation and submission. Reusable functions keep repeated operations consistent.',
 how='The final notebook has 15 cells organized by Markdown headings. M2FeatureBuilder separates fit from transform; record handles repeated fitting/evaluation; lgb_frame and cb_frame prepare model-specific inputs. Comments explain the category vocabulary and missing-value sentinel.',
 read='The comment says the category vocabulary is fitting-only. lgb_frame then applies the supplied levels without learning new ones. cb_frame is a separate function for numeric sentinel handling.',
 result='The original executed archive has seven executed code cells, seven cells with saved outputs and zero error outputs. Successful execution is evidence of a completed run; it does not prove there are no possible bugs.',
 snippets=[(10,55,63),(10,105,109)],outputs=[(8,1),(14,1)],where='features',
 say='I organized the notebook into stages and reused named preprocessing and evaluation functions. I can explain those functions and the comments rather than relying only on formatting.'),
 dict(id='engineering',title='Used feature engineering',
 meaning='Feature engineering turns raw fields into a representation that expresses useful relationships, such as total age, time intervals or multiple facilities per listing.',
 how='Cell 8 derives no-review and anomaly flags, combined age in months, a seats-per-workstation ratio, date intervals and multi-hot facilities. The builder fits its caps and vocabulary only on fitting rows.',
 read='years × 12 + months expresses age in one unit. log1p compresses large nonnegative values. Replacing zero workstations with NaN avoids dividing by zero. Facility tokens become separate indicator columns because a listing can have several facilities.',
 result='The fitting engineered frame has 227 columns and a 144-token facilities vocabulary. The baseline encoded matrix has 1,165 columns; the native LightGBM branch uses the engineered frame with fitted categorical levels.',
 snippets=[(8,96,110),(8,127,149)],outputs=[(8,1)],where='features',
 say='I created time, age, capacity, facilities and anomaly features. I learned the vocabulary and caps on fitting data and reused them on tuning and audit rows.'),
 dict(id='pipelines',title='Used pipelines for preprocessing',
 meaning='A Pipeline runs transformations in sequence. A ColumnTransformer sends different columns through different transformations and combines their results.',
 how='The baseline numeric branch runs median imputation and StandardScaler. The categorical branch runs OneHotEncoder. preprocess.fit_transform(F) learns from fitting rows; preprocess.transform(T) applies the frozen preprocessing to tuning rows.',
 read='The imputer fills missing numbers before the scaler uses them. add_indicator=True adds missingness information where the fitted imputer creates indicators. The category encoder expands nominal values into numeric columns.',
 result='The saved baseline matrix has shape (195842, 1165). SGD and Extra Trees reuse this matrix. LightGBM and CatBoost take their own input paths, so this pipeline should not be described as the final LightGBM’s one-hot input.',
 snippets=[(8,163,184)],outputs=[(8,1)],where='features',
 say='I used a ColumnTransformer containing a numeric Pipeline and a categorical encoder for the baselines. I fitted it on fitting data and transformed the evaluation data without refitting.'),
 dict(id='encoding',title='Encoded categorical features',
 meaning='A nominal category is a name, not a meaningful numeric magnitude. The representation should not accidentally imply that one city code is “twice” another.',
 how='The baseline uses one-hot encoding with an unknown-category policy. LightGBM uses pandas categorical columns with levels learned from fitting rows. Facilities use a separate multi-hot representation.',
 read='OneHotEncoder creates indicator columns. handle_unknown="ignore" handles values absent from its fitted categories. pd.Categorical(..., categories=levels[c]) applies a frozen category list; an unknown value becomes missing for that LightGBM input.',
 result='The 1,165-column baseline matrix records encoded inputs. LightGBM’s category vocabulary is created from F, and lgb_frame applies the same levels to T and A.',
 snippets=[(8,175,184),(10,55,66)],outputs=[(8,1)],where='features',
 say='I used one-hot encoding for the baselines and native categorical inputs for boosting, with fitting-only levels for LightGBM. Facilities were multi-hot because multiple tokens can be present together.'),
 dict(id='missing',title='Identified and handled missing data',
 meaning='A missing value means the value is unavailable. It should not automatically be treated as a real zero. Handling can differ by the estimator and the meaning of the column.',
 how='EDA reports missing fractions. Cleaning converts invalid numeric values and infinities into missing values. The baseline median-imputes numeric fields and adds indicators; categorical missing values get an explicit token. CatBoost’s numeric branch uses a −999 sentinel.',
 read='SimpleImputer learns medians during the fitting stage. fillna("__MISSING__") preserves missing categorical entries as an explicit string. cb_frame changes only its numeric columns; the same sentinel rule is applied consistently to its fit and evaluation inputs.',
 result='The original missingness table identifies borough and review-related columns as important cases. The preprocessing cell produces a fitted baseline matrix and the recorded models complete fitting and prediction.',
 snippets=[(4,5,9),(8,160,171),(10,105,109)],outputs=[(4,2),(8,1)],where='features',
 say='I inspected missingness, separated missing values from genuine zeros, and used fitted imputation for the baselines plus model-specific handling for the boosting branches.'),
 dict(id='findings',title='Highlighted important dataset findings',
 meaning='A finding is a defensible observation connected to evidence. It should explain what we learned and its limits, rather than imply causation from a descriptive plot.',
 how='The notebook shows label mixtures inside metro, space_class and access_type. Later it groups audit errors by metro and space_class, keeping displayed groups with at least 100 rows. It also reports feature-reliance diagnostics.',
 read='normalize="index" makes a category’s row show tier proportions. error_rate is a mean of error booleans. count>=100 restricts the displayed error table; it is not a confidence interval or a proof of why a group is difficult.',
 result='Recorded findings include high borough missingness, duplicate profiles, different tier mixtures by access type and weaker tier-3 performance. zone, metro and snapshot_span_days lead the saved permutation diagnostic; this measures predictive reliance, not causal price effects.',
 snippets=[(4,21,26),(12,32,45)],outputs=[(4,6),(12,4),(12,5),(12,7)],where='audit',
 say='I connected data quality and category patterns to the modelling decisions, and I reported class and subgroup weaknesses. These observations describe this dataset and model; they do not establish causal effects.')
]

PRACTICE = [
 ('Inspect missingness and separate the target', 'Starting with a training DataFrame named train, write the lines to separate predictors and labels, then show the five most-missing columns.',
  'X = train.drop(columns="price_tier")\ny = train["price_tier"].astype(int)\nmissing = X.isna().mean().sort_values(ascending=False)\nprint(missing.head(5))',
  'Explain why price_tier is excluded from X, what mean() does to a Boolean missingness mask, and why a missing fraction of 0.76 means about 76%.'),
 ('Fit preprocessing without leaking evaluation data', 'Assume preprocess is the ColumnTransformer from cell 8 and F, T and A are the engineered fit/tune/audit frames. Write the correct three calls.',
  'Zf = preprocess.fit_transform(F)\nZt = preprocess.transform(T)\nZa = preprocess.transform(A)',
  'Only the first call learns medians, scaling statistics and categories. Calling fit_transform separately on T or A would change those rules using evaluation rows.'),
 ('Predict and calculate macro F1', 'Assume model is a fitted classifier, tune_data is in its required representation, and yt contains tuning labels. Write prediction, macro F1 and accuracy.',
  'from sklearn.metrics import f1_score, accuracy_score\n\nprediction = model.predict(tune_data).reshape(-1).astype(int)\nmacro_f1 = f1_score(yt, prediction, average="macro")\naccuracy = accuracy_score(yt, prediction)\nprint(macro_f1, accuracy)',
  'Explain why macro F1 weights classes equally while accuracy weights rows equally. The final notebook uses different input representations for different model families.'),
 ('Calculate macro F1 from a confusion matrix', 'Write macro F1 without calling sklearn.metrics. Assume matrix is a nonempty square list of counts with true classes as rows and predicted classes as columns.',
  'def macro_f1_from_matrix(matrix):\n    scores = []\n    for k in range(len(matrix)):\n        tp = matrix[k][k]\n        fn = sum(matrix[k]) - tp\n        fp = sum(row[k] for row in matrix) - tp\n        denominator = 2 * tp + fp + fn\n        scores.append(2 * tp / denominator if denominator else 0.0)\n    return sum(scores) / len(scores)',
  'Explain TP, FP and FN, the zero-denominator convention, and why averaging class F1 is different from calculating F1 from average precision and recall.'),
 ('Check that duplicate groups do not cross partitions', 'Assume groups, fit_idx, tune_idx and audit_idx are the objects created in cell 6. Write checks for all three partition pairs.',
  'fit_groups = set(groups.iloc[fit_idx])\ntune_groups = set(groups.iloc[tune_idx])\naudit_groups = set(groups.iloc[audit_idx])\nassert fit_groups.isdisjoint(tune_groups)\nassert fit_groups.isdisjoint(audit_groups)\nassert tune_groups.isdisjoint(audit_groups)',
  'Explain what the profile hash includes and why group disjointness does not prove that different operators or cities are held out.'),
 ('Check submission alignment', 'Assume submission is the final output DataFrame and test is the test table. Write the essential ID, length, valid-class and missing-value checks.',
  'assert submission["ID"].equals(test["space_id"])\nassert len(submission) == len(test)\nassert submission["price_tier"].isin([1, 2, 3, 4]).all()\nassert not submission.isna().any().any()\nsubmission.to_csv("submission.csv", index=False)',
  'Explain why a correct prediction attached to the wrong ID is still an incorrect submission, and why formatting checks cannot establish hidden-label accuracy.')
]

def render_requirements(notebook):
 esc=lambda value:html.escape(str(value),quote=True)
 manifest=[]
 def excerpt(cell,start,end,criterion):
  lines=''.join(notebook['cells'][cell-1]['source']).splitlines()
  assert 1<=start<=end<=len(lines)
  code='\n'.join(lines[start-1:end])
  manifest.append(dict(criterion=criterion,cell=cell,start=start,end=end,code=code))
  rendered=highlight(code,PythonLexer(),HtmlFormatter(linenos='inline',linenostart=start))
  return f'<div class="requirement-source"><div class="requirement-source-head"><span>Original excerpt · cell {cell}, lines {start}–{end}</span><button onclick="goLine({cell},{start})">Find in full notebook ↑</button></div><div class="requirement-code" data-cell="{cell}" data-start="{start}" data-end="{end}">{rendered}</div></div>'
 cards=[]
 for index,c in enumerate(CHECKS,1):
  samples=''.join(excerpt(cell,start,end,c['id']) for cell,start,end in c['snippets'])
  outputs=''.join(f'<a class="button" href="#cell-{cell}-output-{out}">Saved output · cell {cell}, item {out} ↑</a>' for cell,out in c['outputs'])
  cards.append(f'''<article class="requirement-card" id="requirement-{c['id']}"><div class="requirement-heading"><span class="requirement-status">YES · demonstrated in notebook</span>{'<span class="mandatory-badge">L1 mandatory task</span>' if c.get('mandatory') else ''}<h3>{index:02d} · {esc(c['title'])}</h3></div><div class="requirement-explainer"><div><h4>What it means</h4><p>{esc(c['meaning'])}</p></div><div><h4>How we did it</h4><p>{esc(c['how'])}</p></div></div><h4>Read this code like this</h4><p>{esc(c['read'])}</p><p class="caption">The excerpts retain the original Python and line numbers. Some depend on definitions elsewhere in the same cell; use “Find in full notebook” for the surrounding context.</p>{samples}<div class="requirement-proof"><h4>Evidence from the saved run</h4><p>{esc(c['result'])}</p><div class="row">{outputs}</div></div><div class="requirement-viva"><strong>Explain it in the viva</strong><p>{esc(c['say'])}</p><a href="#{c['where']}">Read the full walkthrough ↑</a></div></article>''')
 summary=''.join(f'<a href="#requirement-{c["id"]}"><span>YES</span>{esc(c["title"])}</a>' for c in CHECKS)
 practice=[]
 for i,(title,prompt,solution,explanation) in enumerate(PRACTICE,1):
  compile(solution,f'practice-{i}','exec')
  code=highlight(solution,PythonLexer(),HtmlFormatter())
  practice.append(f'''<article class="coding-practice" id="l1-exercise-{i}"><span class="eyebrow">Coding exercise {i:02d}</span><h4>{esc(title)}</h4><p>{esc(prompt)}</p><details><summary>Show a study solution</summary><div class="requirement-code">{code}</div><p>{esc(explanation)}</p></details></article>''')
 html_text=f'''<section id="requirements" class="requirements-section"><span class="eyebrow">Your declarations, connected to the final notebook</span><h2>Requirements and L1 viva</h2><p>Here is where each “Yes” is supported: the concept, the exact original code, the saved evidence and an explanation you should be able to give yourself. These are study notes, not a submitted form.</p><div class="requirement-summary">{summary}<a href="#authorship-declaration"><span class="no-status">NO</span>“I wrote every line myself”</a></div><div class="l1-boundary"><h3>The L1 criteria you supplied</h3><ul><li><strong>Perform EDA:</strong> demonstrated in cell 4.</li><li><strong>Make predictions using at least three models:</strong> demonstrated in cell 10, with audit predictions in cell 12.</li><li><strong>Explain the notebook and write code when asked:</strong> you must demonstrate this personally to the examiner. Completing the notebook or reading this guide does not establish that skill.</li><li><strong>Pass mark:</strong> according to the rules you supplied, at least <strong>75% in the L1 viva</strong>, in addition to meeting the mandatory tasks. Your Kaggle score does not replace this viva mark.</li></ul><a class="button" href="#l1-code-practice">Practise the coding part ↓</a></div>{''.join(cards)}<article id="authorship-declaration" class="authorship-declaration"><span class="requirement-status no-status">NO · authorship declaration</span><h3>“I have written every line of code myself and have not copied any part of it.”</h3><p>AI assistance was used to create and modify this project’s code, and the feature builder reuses earlier milestone work. That statement should therefore not be marked “Yes.” The technical checklist above describes implemented work; it does not establish sole personal authorship.</p><p>Prepare to explain, modify and write code yourself in the viva. If the course asks about assistance, describe it accurately and follow its instructions.</p></article><div id="l1-code-practice" class="l1-code-practice"><span class="eyebrow">Practise without looking at the answer</span><h3>Write a few lines yourself</h3><p>Try these in a blank editor first, then explain each line out loud. Solutions below are teaching examples with their assumptions stated; they are not new notebook execution outputs. They do not run code or grade your viva readiness in the browser.</p><div class="coding-practice-grid">{''.join(practice)}</div><div class="requirement-proof"><h4>Your readiness check</h4><p>Can you explain why the split happens before fitted preprocessing, write a transform-only evaluation path, distinguish the four model families, calculate macro F1 and verify submission ordering? Test those skills without reading the solutions. The examiner assesses your demonstration.</p><a href="#viva">Practise the full viva questions ↓</a></div></div></section>'''
 return html_text,manifest
