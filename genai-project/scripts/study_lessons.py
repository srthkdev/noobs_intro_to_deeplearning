"""Teaching notes grounded in the archived project; no synthetic results."""
FIGURE_LESSONS = {
'data-summary': {
 'question':'What did we learn before training anything?',
 'read':'There are three panels. Left: defect category on the horizontal axis and number of images containing it on the vertical axis. Middle: how many different classes appear in one image. Right: signed Cohen’s d, a standardized difference between defect-pixel and same-image background brightness.',
 'observed':'Class C occurs in 4,100 images, B in only 313. The middle panel includes 5,516 clean images. The brightness effect for class D is near zero (−0.046); the others are positive on average.',
 'meaning':'The task is imbalanced: learning the common class does not ensure learning the rare one. Brightness alone cannot separate all defects; texture and spatial context matter. The sign of Cohen’s d describes a mean difference, not prediction accuracy.',
 'limit':'Class counts overlap because one image may contain several defects. This is an annotation audit, not model performance.',
 'code':'dataset'},
'defect-examples': {
 'question':'What exactly is a segmentation target?',
 'read':'Each row is one defect class. The left column is a native grayscale image. The right column shows the same image with decoded ground-truth mask pixels painted yellow. Read the mask as “these locations are labelled defective.”',
 'observed':'The labels can cover thin lines or broad irregular patches. Most of each strip remains background, even when a defect is present.',
 'meaning':'Our model must produce a separate yes/no decision for every pixel and class. An image-level label would lose the location and shape. Keeping the long strip geometry avoids stretching these defects into a square.',
 'limit':'These yellow regions are supplied labels decoded from RLE. They are not model predictions and they do not demonstrate model accuracy.',
 'code':'rle'},
 'thresholds': {
 'question':'Why can a more cautious model get a misleadingly better Dice?',
 'read':'All three horizontal axes are the probability threshold: a higher value keeps fewer predicted pixels. Left: three different aggregate scores. Middle: the fraction of empty pairs predicted completely empty. Right: false-positive pixel fraction on empty pairs.',
 'observed':'The blue naive-Dice curve rises at high thresholds, while the orange foreground Dice falls. Empty accuracy rises and false-positive pixels fall. The green weighted score is much less impressed than naive Dice.',
 'meaning':'Erasing predictions helps the many empty pairs but can erase real defects too. That is why we checked foreground quality separately and used the specified weighted assessment score.',
 'limit':'This belongs to the five-epoch M1 baseline. It is not the final ensemble threshold search; the final thresholds are 0.35 for each class.',
 'code':'score'},
 'm2-curves': {
 'question':'Did the five-epoch RGB baseline learn anything?',
 'read':'The horizontal axis is completed epochs 1–5. The vertical axis is loss, so lower is better. The training line measures batches used to update weights; validation measures a held-out loader without weight updates.',
 'observed':'Recorded training loss goes from 0.4844 to 0.3573, and validation loss from 0.4064 to 0.3581.',
 'meaning':'Both losses falling is evidence that this training objective improved on both loaders. To know whether the segmentation is useful, we still need foreground metrics and mask inspection.',
 'limit':'Five points do not establish long-term convergence. M2 uses RGB and the older small U-Net; it is a separate setup from the final grayscale models.',
 'code':'baseline'},
 'm4-curves': {
 'question':'Why did falling loss fail to produce useful defects in M4?',
 'read':'Left: epoch versus official validation score for four controlled runs (R0, R0b, R1, R2). Right: epoch versus training and validation loss for the displayed run. Lower loss is better; higher score is better.',
 'observed':'The scores approach about 0.30. The saved foreground Dice is zero for these short runs. Loss still falls over six epochs.',
 'meaning':'A score near the all-empty baseline and zero foreground Dice means the model is not segmenting the positive regions usefully at evaluation. It can improve background decisions without learning the foreground well. Calling this successful foreground learning would be wrong.',
 'limit':'These runs used a 1,200-image subset and six epochs. They do not measure the final full-data, 25-epoch models.',
 'code':'m4-analysis'},
 'class-balance': {
 'question':'Which classes need special attention?',
 'read':'Each bar is a defect category. Its height counts images containing that category, rather than pixels or mutually exclusive labels.',
 'observed':'A: 1,081 images; B: 313; C: 4,100; D: 1,707. B has far fewer examples than C.',
 'meaning':'We should inspect per-class results. A single average can hide weak rare-class performance. The final training sampler increases exposure to rare positive classes with tempered weights.',
 'limit':'This report figure restates the same full-dataset prevalence as the EDA. It is not independent confirmation that the sampler improved a score.',
 'code':'dataset'},
 'scratch-architecture': {
 'question':'What did the original report diagram try to summarize?',
 'read':'Follow the arrows from image to encoder, then decoder and segmentation head. Encoder features become spatially smaller; the decoder brings predictions back to the original resolution.',
 'observed':'The report contains this compact summary. The new interactive architecture above expands the actual stage shapes and skip paths verified from the saved model.',
 'meaning':'A diagram is a map of computation. It is useful when you can point from a box to the corresponding operation in code and explain its input and output.',
 'limit':'This is a structural summary, not a plot of learned features or an execution trace. Use the exact interactive shapes for detailed questions.',
 'code':'custom-model'},
 'final-curves': {
 'question':'Why did we save an earlier EfficientNet checkpoint?',
 'read':'One panel per architecture, each with epoch on the horizontal axis and loss on the vertical axis. Blue is training loss; orange is tune loss. Compare the trend within a panel before comparing models.',
 'observed':'Training losses continue falling. EfficientNet’s tune loss reaches its lowest recorded value at epoch 19 (0.19437), then ends slightly higher at epoch 25 (0.19653). The other two selected checkpoints are epoch 25.',
 'meaning':'Later training can fit the training data better without improving the held-out objective. We saved checkpoints using tune loss, so the EfficientNet checkpoint comes from epoch 19. A widening gap suggests overfitting to the training objective, but does not by itself diagnose every cause.',
 'limit':'Loss is not the public Kaggle score. The lines are recorded runs, not uncertainty intervals. We did not claim that training longer must improve the competition result.',
 'code':'loss'},
 'errors': {
 'question':'What failures are hidden by a single final score?',
 'read':'Each row compares one image/class pair. Left: red ground-truth regions. Right: red predicted regions. The Dice label refers to that pair only. Look for missing true regions and added red regions in the wrong place.',
 'observed':'These examples were selected as failures. Some predictions miss the labelled region; the bottom class-C example adds a large false-positive region. The displayed pairs have Dice 0.',
 'meaning':'A false negative misses a real defect. A false positive flags good material. We need both kinds of inspection because a factory pays for both missed defects and unnecessary rejection.',
 'limit':'This is a deliberately selected error gallery, not a random sample or an estimate that every prediction fails. Inspect the successful real-mask examples too.',
 'code':'components'},
}

# Each step highlights the corresponding node in a newly drawn checkpoint-derived SVG.
MODEL_LESSONS = {
 'scratch_unet': [
 ('input','1. Start with one strip','1 × 128 × 800 means one intensity channel, 128 rows and 800 columns. We usually add a batch axis B in code. After /255 and (x−0.5)/0.25, the same pixel locations are represented by normalized numbers.','dataset'),
 ('enc0','2. Learn local features','The first block produces 24 × 128 × 800. Each of the 24 channels is a learned feature map, not a defect category. Two Conv → BatchNorm → ReLU sequences keep the spatial size here.','custom-model'),
 ('enc1','3. Trade resolution for context','MaxPool halves both spatial dimensions. The next block yields 48 × 64 × 400. A feature now summarizes a larger region of the original strip; some precise location detail has been discarded.','custom-model'),
 ('enc2','4. Go deeper','Another pool and block gives 96 × 32 × 200. The network combines simpler patterns into more useful representations. Calling every channel a known “scratch detector” would be an interpretation we have not verified.','custom-model'),
 ('enc3','5. Reach the bottom of the U','The deepest block is 192 × 16 × 100. It has more channels but fewer spatial positions. There is no separate fifth bottleneck in this custom implementation.','custom-model'),
 ('dec0','6. Restore detail with a skip','Upsample 192 channels to 32 × 200, then concatenate the matching 96-channel encoder feature. 192 + 96 = 288 channels enter the decoder block; 96 come out. Concatenation stacks channels; it does not add their values.','custom-model'),
 ('dec2','7. Return to full resolution','The remaining decoder stages combine 96+48 channels, then 48+24 channels. They output 48 × 64 × 400 and 24 × 128 × 800. Long orange arrows carry earlier spatial detail straight into the decoder.','custom-model'),
 ('head','8. Turn features into class probabilities','The 1×1 head maps 24 features to four logits at every original pixel. Sigmoid converts each logit independently to a probability. These are still soft maps; ensemble averaging, thresholding and area filtering happen later.','prediction')],
 'resnet34_unet': [
 ('input','1. Give a pretrained encoder our strip','Input is still B × 1 × 128 × 800. SMP adapts the first RGB convolution to one channel by summing the three pretrained input-weight slices. We fine-tune the encoder rather than freezing it.','model-definition'),
 ('enc0','2. The first downsampling happens early','The first recorded encoder feature is 64 × 64 × 400. Unlike the custom U-Net, this ResNet stem already halves the image resolution. The original input is also returned in the encoder feature list, but is not a decoder skip here.','model-definition'),
 ('enc1','3. Enter residual stages','The next feature is 64 × 32 × 200. Inside ResNet blocks, a shortcut adds a block input to a learned transformation, with a projection when needed. Those short additions are different from the long concatenating U-Net skips drawn here.','model-definition'),
 ('enc3','4. Extract coarser representations','The next two outputs are 128 × 16 × 100 and 256 × 8 × 50. ImageNet weights give an initial feature representation; training on masks adapts it to this industrial task. Pretraining does not guarantee success.','model-definition'),
 ('enc4','5. Reach the deepest feature','The deepest encoder output is 512 × 4 × 25. Downsampling by 32 in each direction explains 128/32 = 4 and 800/32 = 25. These are features, not a tiny four-class mask.','model-definition'),
 ('dec0','6. Start the U-Net decoder','The first decoder stage upsamples and concatenates the 256 × 8 × 50 encoder skip. Its output is 256 × 8 × 50. The next decoder outputs are 128 × 16 × 100, 64 × 32 × 200 and 32 × 64 × 400.','model-definition'),
 ('dec4','7. Finish without an extra skip','The fifth decoder stage restores 16 × 128 × 800. There are five decoder stages but four encoder skips in this configuration. The last stage has no encoder skip.','model-definition'),
 ('head','8. Read the output in the same way','The head produces four full-resolution logits and sigmoid provides four independent probabilities. A bigger encoder changes how features are obtained; it does not change the required output format.','prediction')],
 'efficientnet_b0_fpn': [
 ('input','1. Separate encoder from decoder','EfficientNet-B0 is the pretrained feature extractor; FPN is the decoder. The encoder’s original classification head is not our four-mask output. Our input remains one normalized grayscale strip.','model-definition'),
 ('enc0','2. Keep several scales','The FPN takes the last four encoder outputs: 24 × 32 × 200, 40 × 16 × 100, 112 × 8 × 50 and 320 × 4 × 25. The earlier stem output is 32 × 64 × 400 and is not one of these four pyramid inputs.','model-definition'),
 ('p5','3. Project the deepest scale','A 1×1 projection changes 320 channels to 128 while keeping 4 × 25 positions. This is P5. Matching channel counts makes element-wise addition possible later.','model-definition'),
 ('p4','4. Add lateral and top-down information','Upsample P5 to 8 × 50 and add a 128-channel projection of the matching encoder feature. That creates P4. The + circle means values at matching positions are added; the channel count stays 128.','model-definition'),
 ('p2','5. Repeat toward finer resolution','The same idea produces P3 at 16 × 100 and P2 at 32 × 200, all with 128 channels. Fine levels receive information propagated from deeper levels without discarding their own spatial detail.','model-definition'),
 ('align','6. Put all scales on one grid','Segmentation blocks convert the four pyramid outputs to 64 × 32 × 200. This includes upsampling the coarser scales. The arrows converge because the outputs now have matching shapes.','model-definition'),
 ('merge','7. Add the aligned features','This configuration adds the four aligned 64-channel tensors. It does not concatenate them to 256 channels. Dropout2d is used in the decoder; evaluation mode disables its random dropping.','model-definition'),
 ('head','8. Predict at the original size','The class head and 4× upsampling produce four logits at 128 × 800. Sigmoid is applied outside the model. FPN fuses scales using addition; our U-Net decoders mainly use channel concatenation.','prediction')]
}

NOTEBOOK_GUIDES = {
 'milestone-1':'Read the full-dataset audit first, then inspect the class counts, decoded masks and five-epoch log. The threshold table explains why empty pairs can inflate naive Dice. This archive contains actual outputs and attached figures.',
 'milestone-2':'The co-occurrence table counts images with both labels; the correlation table measures association between presence indicators. The diagonal of the count table is class prevalence. The five-epoch RGB model is a baseline, separate from the final grayscale models.',
 'milestone-4':'This file defines the controlled experiments. It has no saved cell outputs. The original results and plot are shown below it as separate artifacts; they show foreground underfitting, not a successful final model.',
 'milestone-5':'This notebook contains the post-processing exercises. Its outputs were saved in answers.json rather than in notebook cells. The original answers are shown below; the toy strict-threshold and 8-connectivity settings differ from the final inference implementation.',
 'final-training':'The long training cell contains the actual CUDA run log for all three models. Scroll past installation and download messages to the epoch records. The final artifact section reorganizes the saved histories into readable tables; it is not a new training run.',
 'final-validation-source':'This is source without saved cell outputs. The comparison tables and failure gallery below are separately recorded validation artifacts. The final ensemble was selected on tune, even though horizontal TTA scored higher on audit.',
 'final-inference':'This executed archive shows the frozen inference code and submission preview. The saved output records 3,000 test images, 12,000 image-class rows and 4,370 nonempty masks. A preview table is evidence of output formatting, not hidden-test correctness.',
 'shared-inference-code':'This companion explains the deployed ONNX browser inference. It is a source archive without saved notebook outputs. The deployment check shown below is separate evidence from the real demo verification.'
}
