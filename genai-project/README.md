# Surface defect segmentation — viva study guide

[Open the interactive guide](study-guide/genai-viva-guide.html) or download it to read offline. The HTML embeds its diagrams, images, notebook viewers and source downloads. Desktop browsers are the primary target.

The guide contains 50 concept explanations, 30 viva questions, eight real notebooks, checkpoint-verified architecture diagrams, training histories, evaluation results and interactive mask examples. Original notebooks, Python code, reports and figures are included here and in `study-guide/genai-source-evidence.zip`.

Evidence is recorded from the actual project. W&B verification timestamps distinguish live verification from archived history; the first milestone's live run could not be fetched. Leaderboard ranks are historical. Three CPU checkpoint replays reproduce all 12 recorded sample/class pixel counts. No checkpoint weights or access credentials are distributed.

`study-guide/guide-template.html`, `guide.css`, `guide.js`, and `scripts/viva_content.py` hold the guide source. The builder originally ran in the complete training repository; rerunning evidence preparation requires its data, checkpoint artifacts and vendor wheel, which are intentionally absent from this public study package. The generated HTML needs none of those dependencies.
