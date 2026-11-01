# MLP final notebook — single-page viva guide

**[Read online](https://srthkdev.github.io/noobs_intro_to_deeplearning/mlp/)**

The complete executed final notebook is the centre of this continuous HTML page. All 15 original cells, seven code cells and 25 saved output items appear in order. Each code cell has what/why/how notes, relevant concepts, original-line jump links, output explanations and a viva answer. Code is visible by default; long cells have scroll panels and an expand control.

The page also includes 40 linked concepts, 35 viva questions, original EDA/diagnostic plots, the recorded five-configuration comparison and an interactive confusion matrix. Its exact aggregate counts reproduce the recorded audit macro F1. Scores distinguish tuning, audit and the actual registered version-3 Kaggle code submission.

`project/notebooks/final.ipynb` is copied byte for byte from the executed archive. `project/reports/final-results/` contains the original final results. No fitting or notebook execution is performed to build the guide. The full audit-predictions CSV is not distributed; `evidence/audit-confusion.json` contains aggregate counts and the source file hash.

The generated `index.html` embeds its styles, scripts and original images. It works offline when opened directly. External documentation/Kaggle links and separate source-file links require their destinations, but the entire study content is already on the page.

Rebuild from the repository root:

```sh
python3 -m venv .venv-guide
.venv-guide/bin/pip install -r mlp/requirements.txt
.venv-guide/bin/python mlp/build_page.py
```

- `content.py`: cell explanations, concept glossary and viva questions.
- `template.html`, `style.css`, `script.js`: continuous-page design and interactions.
- `build_page.py`: static Jupyter nbconvert rendering and original-source checks.
- `evidence/source-manifest.json`, `render-verification.json`: provenance and cell/output checks.
- `evidence/browser-verification.json`: desktop browser, interaction and offline checks.

The initial EDA used the full labelled table. The guide therefore describes audit withholding accurately as applying to subsequent fitted preprocessing, training, stopping and winner selection. The final grouped holdout is distinct from the separate M3/M4 milestone studies.
