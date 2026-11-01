# Surface defect segmentation — viva study guide

[Open the interactive guide](study-guide/genai-viva-guide.html) or download it to read offline. The HTML embeds its diagrams, images, notebook viewers and source downloads. Desktop browsers are the primary target.

The guide contains 50 concept explanations, 30 viva questions, eight real notebooks, checkpoint-verified architecture diagrams, training histories, evaluation results and interactive mask examples. Original notebooks, Python code, reports and figures are included here and in `study-guide/genai-source-evidence.zip`.

Evidence is recorded from the actual project. W&B verification timestamps distinguish live verification from archived history; the first milestone's live run could not be fetched. Leaderboard ranks are historical. Three CPU checkpoint replays reproduce all 12 recorded sample/class pixel counts. No checkpoint weights or access credentials are distributed.

`study-guide/guide-template.html`, `guide.css`, `guide.js`, and `scripts/viva_content.py` hold the guide source. The builder originally ran in the complete training repository; rerunning evidence preparation requires its data, checkpoint artifacts and vendor wheel, which are intentionally absent from this public study package. The generated HTML needs none of those dependencies.

## Browser study experience

The notebook chapter uses Jupyter nbconvert to display all eight original notebooks on the page, including Markdown, highlighted code, stored outputs, dataframe tables and attached figures. Its outline and code/output filters help navigate long cells. Source-only notebooks show their separately saved results in a labelled section below them. Complete Python modules and commented code walkthroughs are also readable directly in the guide.

The model chapter starts with attributed original U-Net, residual-block and FPN reference figures, explains how to read them, and contrasts each with this project. The new vector diagrams have eight guided steps per model, exact checkpoint-verified tensor shapes, click-to-explain stages and zoom controls. Every original EDA/training/error figure has notes explaining its axes, recorded observations, implications and limits.

Rebuild this public package without training data or checkpoints:

```sh
python3 -m venv .venv-guide
.venv-guide/bin/pip install -r genai-project/scripts/requirements-study.txt
.venv-guide/bin/python genai-project/scripts/build_study_site.py
```

The compressed evidence snapshot preserves the original artifact payload. The public builder renders unchanged notebook bytes, updates derived diagrams and regenerates the self-contained HTML and download bundle. It never executes notebook cells or fetches account data. Research figure attribution and rendering sources are in [references/SOURCES.md](study-guide/references/SOURCES.md).
