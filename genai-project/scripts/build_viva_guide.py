"""Assemble a self-contained HTML study guide from actual repository evidence."""
from pathlib import Path
import ast,base64,csv,hashlib,html,io,json,mimetypes,re,subprocess,zipfile,datetime
import pandas as pd
from html.parser import HTMLParser
from viva_content import CONCEPTS,QUIZ,MILESTONES
R=Path(__file__).resolve().parents[1];S=R/'study-guide';E=S/'evidence';D=S/'diagrams';D.mkdir(exist_ok=True)
readj=lambda p:json.loads((R/p).read_text())
b64=lambda b:base64.b64encode(b).decode()
esc=lambda s:html.escape(str(s),quote=True)
shapes=readj('study-guide/evidence/architectures.json')
# Diagram nodes correspond to verified stage outputs; full module text is attached separately.
def svg_start(w=1120,h=650,title='Architecture'):
 return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="group" aria-label="{esc(title)}"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#668177"/></marker></defs><style>text{{font-family:Arial,sans-serif;fill:#263f38}}.title{{font-size:15px;font-weight:bold}}.sub{{font-size:12px;fill:#577168}}.tiny{{font-size:11px;fill:#6c8077}}.wire{{fill:none;stroke:#668177;stroke-width:1.7;marker-end:url(#arrow)}}.skip{{fill:none;stroke:#be7951;stroke-width:1.7;stroke-dasharray:6 4;marker-end:url(#arrow)}}</style><title>{esc(title)}</title>'
def box(x,y,w,label,sub,key='',fill='#e7f1eb',h=62):
 attrs=f'data-detail="{esc(key)}" tabindex="0" role="button" aria-label="{esc(label+": "+sub)}"' if key else ''
 return f'<g {attrs}><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="#bdcec2"/><text x="{x+13}" y="{y+24}" class="title">{esc(label)}</text><text x="{x+13}" y="{y+45}" class="sub">{esc(sub)}</text></g>'
def path(d,skip=False):return f'<path d="{d}" class="{"skip" if skip else "wire"}"/>'
def text(x,y,t,cls='tiny'):return f'<text x="{x}" y="{y}" class="{cls}">{esc(t)}</text>'
def cshape(v):return ' × '.join(map(str,v[1:]))
svgs={};details={}
for record in shapes:
 name=record['name'];st=record['stages'];z=[svg_start(title=name+' · actual checkpoint stage shapes')]
 if name=='scratch_unet':
  z+=[text(35,28,'ENCODER · context', 'title'),text(590,28,'DECODER · localization','title')]
  for i,y in enumerate([65,190,315,440]):
   shape=st[f'enc.{i}']['output'];key=f'{name}:enc{i}';z.append(box(35,y,275,f'Encoder block {i+1}',cshape(shape),key));details[key]=f'<strong>Encoder block {i+1}: {cshape(shape)}</strong><p>Two 3×3 convolutions, each followed by BatchNorm and ReLU. '+('The input has one grayscale channel.' if i==0 else 'A 2×2 max pool precedes this block, reducing spatial resolution.')+'</p><p>Exact module: <code>enc.'+str(i)+'</code> in <code>ScratchUNet</code>.</p>'
   if i<3:z+=[path(f'M172 {y+62} V{y+125}'),text(183,y+100,'2×2 max pool')]
  for i,y in enumerate([315,190,65]):
   inp=st[f'dec.{i}']['input'][0];out=st[f'dec.{i}']['output'];key=f'{name}:dec{i}';z.append(box(590,y,275,f'Decoder block {i+1}',cshape(out),key,'#e3edf0'));details[key]=f'<strong>Decoder block {i+1}: {cshape(out)}</strong><p>Bilinearly upsample to the encoder skip size, concatenate along channels, then apply two convolution/BatchNorm/ReLU layers. The actual concatenated input is <code>{cshape(inp)}</code>.</p>'
   z+=[path(f'M310 {y+31} H590',True),text(365,y+22,'concatenate encoder skip')]
   if i<2:z+=[path(f'M727 {y} V{y-63}'),text(742,y-30,'upsample')]
  z+=[path('M310 471 H470 V346 H590'),text(390,454,'upsample'),path('M865 96 H900'),box(900,65,185,'1×1 class head','4 × 128 × 800','logits','#f8eadc'),path('M992 127 V190'),box(900,190,185,'Sigmoid','four probabilities','sigmoid','#f8eadc'),text(35,580,'Input: B × 1 × 128 × 800. Every node shape above was verified after loading the saved trained weights.'),text(35,604,'Orange dashed lines are channel concatenations. The sigmoid is applied after the model returns logits.')]
 elif name=='resnet34_unet':
  z+=[text(35,28,'RESNET34 ENCODER','title'),text(590,28,'U-NET DECODER','title')]
  feat=st['encoder']['output']
  for i,y in enumerate([115,215,315,415,515],1):
   key=f'{name}:enc{i}';z.append(box(35,y,275,f'Encoder feature {i}',cshape(feat[i]),key));details[key]=f'<strong>ResNet34 encoder feature {i}: {cshape(feat[i])}</strong><p>ImageNet-pretrained residual encoder adapted to one grayscale input channel and fine-tuned for segmentation. This is an observed encoder output from the actual loaded model.</p>'
   if i<5:z.append(path(f'M172 {y+62} V{y+100}'))
  for i,y in enumerate([415,315,215,115,40]):
   out=st[f'decoder.blocks.{i}']['output'];key=f'{name}:dec{i}';z.append(box(590,y,275,f'Decoder block {i+1}',cshape(out),key,'#e3edf0'));details[key]=f'<strong>U-Net decoder block {i+1}: {cshape(out)}</strong><p>Upsample the preceding representation and apply the library decoder convolutions. '+('This stage concatenates the matching encoder feature map.' if i<4 else 'The final full-resolution stage has no encoder skip.')+'</p>'
   if i<4:z.append(path(f'M310 {y+31} H590',True))
   if i<3:z.append(path(f'M727 {y} V{y-38}'))
   if i==3:z.append(path('M727 115 V102'))
  z+=[path('M310 546 H470 V446 H590'),path('M865 71 H900'),box(900,40,185,'Class head','4 × 128 × 800','logits','#f8eadc'),path('M992 102 V170'),box(900,170,185,'Sigmoid','four probabilities','sigmoid','#f8eadc'),text(35,626,'Full encoder output list also includes the original input (1 × 128 × 800). Four long skips feed five decoder stages.')]
 else:
  z+=[text(30,28,'EFFICIENTNET-B0 · last four scales','title'),text(385,28,'TOP-DOWN PYRAMID','title'),text(800,28,'ALIGN + MERGE','title')]
  feat=st['encoder']['output']
  for i,(stage,y) in enumerate([(2,90),(3,220),(4,350),(5,480)]):
   key=f'{name}:enc{stage}';z.append(box(30,y,280,f'Encoder feature {stage}',cshape(feat[stage]),key));details[key]=f'<strong>EfficientNet-B0 feature {stage}: {cshape(feat[stage])}</strong><p>The pretrained encoder extracts this scale. The FPN uses these last four scales; the earlier stem feature is 32 × 64 × 400.</p>'
   if i<3:z.append(path(f'M170 {y+62} V{y+130}'))
  for level,y in [(2,90),(3,220),(4,350),(5,480)]:
   out=st[f'decoder.p{level}']['output'];key=f'{name}:p{level}';z.append(box(385,y,270,f'P{level}',cshape(out),key,'#e3edf0'));details[key]=f'<strong>P{level}: {cshape(out)}</strong><p>'+('A 1×1 projection converts the deepest 320-channel encoder feature into 128 pyramid channels.' if level==5 else 'Upsample the deeper pyramid feature and add a projected lateral encoder feature. This is addition after matching channels, not U-Net concatenation.')+'</p>'
   z.append(path(f'M310 {y+31} H385',True))
   if level>2:z.append(path(f'M520 {y} V{y-68}'))
   z.append(path(f'M655 {y+31} C720 {y+31},735 270,800 270'))
  z+=[box(800,240,280,'Add aligned scale features','64 × 32 × 200','fpn-merge','#f8eadc'),path('M940 302 V370'),box(800,370,280,'Class head + 4× upsample','4 × 128 × 800','logits','#f8eadc'),path('M940 432 V500'),box(800,500,280,'Sigmoid','four probabilities','sigmoid','#f8eadc'),text(30,601,'Each pyramid level is converted to 64 channels at 32 × 200 before addition. Decoder also contains Dropout2d.'),text(30,624,'The stage diagram summarizes the graph. Download the full module architecture for every layer.')]
 z.append('</svg>');svgs[name]=''.join(z);(D/(name+'.svg')).write_text(svgs[name])
details.update({'logits':'<strong>Four full-resolution logits</strong><p>The segmentation model returns B × 4 × 128 × 800 unrestricted scores. The head turns decoded features into one logit per defect per pixel. Sigmoid is applied in the probability/inference step.</p>','sigmoid':'<strong>Sigmoid probabilities</strong><p>Each defect channel is mapped independently to [0,1]. The four probabilities do not have to sum to one. Thresholding comes later, after any aligned view/model averaging.</p>','fpn-merge':'<strong>Merge the four pyramid scales</strong><p>Segmentation blocks map each pyramid level to 64 × 32 × 200. This configured FPN adds those aligned features, applies its dropout stage, then uses the segmentation head and final upsampling.</p>'})
pipeline_nodes=[('image','Native grayscale','1 × 128 × 800'),('normalize','Preprocess','/255; (x − 0.5) / 0.25'),('models','Three trained models','four logits per pixel'),('probability','Sigmoid + cache precision','per-model float16 maps'),('mean','Equal probability mean','float32 sum → float16'),('threshold','Class thresholds','0.35 each; p ≥ T'),('area','Component filtering','4-connected; class areas'),('rle','Column-major RLE','preserve image/class IDs')]
pipeline_details={
'image':'Load the original 128-high, 800-wide image as grayscale. The final Dataset does not resize it into a square.',
'normalize':'Convert intensities to [0,1], then apply (x−0.5)/0.25. Training and final prediction use the same rule.',
'models':'Run custom U-Net, ResNet34 U-Net and EfficientNet-B0 FPN with their saved best checkpoints in evaluation mode. All return logits with four channels.',
'probability':'Apply sigmoid to each model output. The final wrapper reproduces validation cache precision by converting each model probability map to float16.',
'mean':'Average the three probability maps equally, summing in float32 and casting the result to float16. The selected final configuration does not use TTA.',
 'threshold':'Compare each class probability using >= its frozen threshold of 0.35. This is deliberately different from strict > in M4/M5 calculation exercises.',
 'area':'Label four-connected components and keep areas at least [32,128,128,128] for classes a–d. Thresholding must precede binary component filtering.',
 'rle':'Encode each final mask with 1-based column-major runs. Write a single space for empty masks and preserve the sample submission IDs and row order.'}
z=[svg_start(980,370,'Exact final inference pipeline')]
for i,(key,label,sub) in enumerate(pipeline_nodes):
 x=[20,260,500,740,740,500,260,20][i];y=45 if i<4 else 240;z.append(box(x,y,215,label,sub,'pipe:'+key,h=70))
 if i<3:z.append(path(f'M{x+215} {y+35} H{x+240}'))
 elif i==3:z.append(path('M847 115 V240'))
 elif i<7:z.append(path(f'M{x} {y+35} H{x-25}'))
z.append('</svg>');pipeline_svg=''.join(z);(D/'final-pipeline.svg').write_text(pipeline_svg)
split=pd.read_csv(R/'reports/final-results/split.csv').partition.value_counts().to_dict();assert split=={'train':9114,'audit':1140,'tune':1139}
z=[svg_start(960,200,'Actual train tune audit partition')];xx=20
for label,width,fill in [('train',720,'#d5e8d9'),('tune',90,'#dbe9ef'),('audit',90,'#f3dfd1')]:
 z.append(f'<rect x="{xx}" y="50" width="{width}" height="50" rx="4" fill="{fill}"/>');z.append(text(xx+width/2-18,80,'80%' if label=='train' else '10%','sub'));xx+=width
z += [text(20,27,'11,393 labelled images · split by image ID, seed 42','title'),text(20,132,'Train: 9,114','title'),text(590,132,'Tune: 1,139','title'),text(770,132,'Audit: 1,140','title'),text(20,170,'The 2,279-image outer holdout is split with seed 43. Rounding makes tune and audit differ by one image.'),'</svg>'];split_svg=''.join(z);(D/'data-split.svg').write_text(split_svg)
# Exact source excerpts and full downloadable source files.
def excerpt(path,name):
 source=(R/path).read_text();tree=ast.parse(source);node=next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name==name)
 return {'path':path,'start':node.lineno,'end':node.end_lineno,'code':'\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])}
snippets={
'dataset':('Load one strip and four masks','src/final_pipeline.py','StripDataset'),
'model-definition':('Construct the three architectures','src/final_pipeline.py','make_model'),
'custom-model':('Follow the custom U-Net','src/final_pipeline.py','ScratchUNet'),
'loss':('BCE plus positive-pair soft Dice','src/final_pipeline.py','loss_fn'),
'components':('Actual four-connected area filtering','src/final_pipeline.py','remove_components'),
'rle':('Actual RLE encoding','src/final_pipeline.py','encode_rle'),
'score':('Per-class metric accumulation','src/final_pipeline.py','score_predictions'),
'prediction':('Frozen final inference and precision','src/predict_final.py','predict_batch'),
'baseline':('The original small U-Net','src/models.py','SmallUNet'),
'm4-analysis':('M4 controlled subset analysis','scripts/milestone4_experiment.py','analyze'),
'm5-calculations':('M5 exact calculations','scripts/milestone5_calculations.py','main')}
source_entries={k:dict(title=title,**excerpt(path,name)) for k,(title,path,name) in snippets.items()}
with zipfile.ZipFile(R.parent/'final-training/vendor/segmentation_models_pytorch-0.5.0-py3-none-any.whl') as wheel:
    utility=wheel.read('segmentation_models_pytorch/encoders/_utils.py')
(E/'smp-0.5.0-encoder-utils.py').write_bytes(utility)
source_entries['adaptation']=dict(title='Actual SMP 0.5.0 first-convolution adaptation',**excerpt('study-guide/evidence/smp-0.5.0-encoder-utils.py','patch_first_conv'))
source_lines=(R/'src/final_pipeline.py').read_text().splitlines()
sa=next(i for i,l in enumerate(source_lines) if l.strip()=='train_ids, holdout_ids = train_test_split(')
sb=next(i for i in range(sa,len(source_lines)) if source_lines[i].strip().startswith('training = StripDataset'))
source_entries['split']={'title':'The saved image-ID split','path':'src/final_pipeline.py','start':sa+1,'end':sb,'code':'\n'.join(source_lines[sa:sb])}

src=(R/'src/final_pipeline.py').read_text().splitlines();a=next(i for i,l in enumerate(src) if l.strip()=='for epoch in range(1, args.epochs + 1):');b=next(i for i in range(a,len(src)) if src[i].strip()=='checkpoint = torch.load(')
source_entries['training-loop']={'title':'The real training loop and checkpoint rule','path':'src/final_pipeline.py','start':a+1,'end':b,'code':'\n'.join(src[a:b])}
# Preserve original figures byte-for-byte in the guide and source bundle.
fig_specs=[('data-summary','reports/figures/data-summary.png','Original M1 data overview'),('defect-examples','reports/figures/defect-examples.png','Original annotated strip examples'),('thresholds','reports/milestone-1-threshold_sweep.png','Original M1 threshold sweep'),('m2-curves','reports/milestone2-results/loss_curves.png','Original five-epoch RGB baseline losses'),('m4-curves','reports/milestone4-results/training-curves.png','Original four-run M4 comparison'),('class-balance','reports/figures-final/class-balance.png','Original final-report class distribution'),('scratch-architecture','reports/figures-final/scratch-architecture.png','Original final-report scratch architecture figure'),('final-curves','reports/figures-final/training-curves.png','Original final-report training curves'),('errors','reports/final-results/inference/error-analysis.png','Original frozen-ensemble audit error analysis')]
figures=[{'id':key,'path':path,'title':title,'data':'data:image/png;base64,'+b64((R/path).read_bytes()),'sha256':hashlib.sha256((R/path).read_bytes()).hexdigest()} for key,path,title in fig_specs]
# Download bytes remain unchanged. No credentials are embedded.
nb_specs=[('notebooks/milestone-1.ipynb','M1 · data and baseline'),('notebooks/milestone-2.ipynb','M2 · RGB Dataset and five-epoch baseline'),('notebooks/milestone-4.ipynb','M4 · controlled four-run experiment'),('notebooks/milestone-5.ipynb','M5 · post-processing calculations'),('notebooks/final-training.ipynb','Final · three-model training'),('notebooks/final-validation-source.ipynb','Final · inference-variant validation source'),('notebooks/final-inference.ipynb','Final · registered inference, executed archive'),('notebooks/shared-inference-code.ipynb','Final · shared inference-code companion')]
files={};manifest=[]
def addfile(path):
 if path in files:return
 raw=(R/path).read_bytes();mime=mimetypes.guess_type(path)[0] or 'application/octet-stream';files[path]={'b64':b64(raw),'mime':mime,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};manifest.append({'path':path,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
for path,_ in nb_specs:addfile(path)
for path in ['src/data.py','src/models.py','src/metrics.py','src/train.py','src/inference.py','src/final_pipeline.py','src/final_validation.py','src/predict_final.py','scripts/milestone4_experiment.py','scripts/milestone5_calculations.py','scripts/interpret_milestone4.py','reports/final-report.pdf','reports/milestones-4-5/23f3000839_genai_M4_M5_worked_answers.docx','reports/milestones-4-5/23f3000839_genai_M4_M5_worked_answers.pdf','reports/final-results/model_comparison.csv','reports/final-results/inference/inference_comparison.csv','reports/final-results/inference/final_selection.json','reports/final-results/kaggle-result.json','reports/final-results/deployment-result.json','reports/training-protocol.txt','study-guide/evidence/architectures.json','study-guide/evidence/wandb-live-verification.json']:
 addfile(path)
for p in sorted((R/'reports/final-results').glob('*/history.csv')):addfile(str(p.relative_to(R)))
for p in sorted((R/'reports/milestone4-results').glob('*/history.csv')):addfile(str(p.relative_to(R)))
for path in ['reports/milestone-1-training_history.csv','reports/milestone2-results/history.csv','reports/milestone4-results/answers.json','reports/milestone5-results/answers.json']:addfile(path)
addfile('study-guide/evidence/smp-0.5.0-encoder-utils.py')
for record in shapes:addfile('study-guide/evidence/'+record['architecture_text'])
for p in sorted(D.glob('*.svg')):addfile(str(p.relative_to(R)))
for _,path,_ in fig_specs:addfile(path)
secret_pattern=r'wandb_v1_[A-Za-z0-9]+|hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+'
for path in files:
 if Path(path).suffix in ['.py','.ipynb','.json'] and re.search(secret_pattern,(R/path).read_text()):raise ValueError('Credential-like literal in '+path)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
(E/'source-manifest.json').write_text(json.dumps({'source_commit':commit,'built_at_utc':stamp,'files':manifest,'note':'Original file hashes; this manifest excludes its own hash. Checkpoint binaries are identified by hashes in architectures.json and are not embedded.'},indent=2)+'\n');addfile('study-guide/evidence/source-manifest.json')
# Notebook output rendering is constrained to text, inline PNGs and sanitized table HTML.
def clean_html(s):
 class Safe(HTMLParser):
  allowed={'table','thead','tbody','tr','th','td','p','div','span','b','strong','i','em','code','pre','ul','ol','li','br','h1','h2','h3','h4','a'}
  forbidden={'script','style','iframe','object','embed'}
  def __init__(self):super().__init__(convert_charrefs=True);self.out=[];self.skip=0
  def handle_starttag(self,tag,attrs):
   if tag in self.forbidden:self.skip+=1;return
   if self.skip or tag not in self.allowed:return
   keep=[]
   for k,v in attrs:
    if (k in ['colspan','rowspan'] and str(v).isdigit()) or (k=='href' and str(v).startswith(('https://','http://','#'))):keep.append(' '+k+'="'+esc(v)+'"')
   self.out.append('<'+tag+''.join(keep)+'>')
  def handle_endtag(self,tag):
   if self.skip:
    if tag in self.forbidden:self.skip-=1
    return
   if tag in self.allowed and tag!='br':self.out.append('</'+tag+'>')
  def handle_data(self,data):
   if not self.skip:self.out.append(html.escape(data))
 parser=Safe();parser.feed(s);return ''.join(parser.out)
notebooks=[]
for path,title in nb_specs:
 nb=readj(path);cells=[];errors=0;output_cells=0
 for c in nb['cells']:
  source=''.join(c.get('source',[])) if isinstance(c.get('source'),list) else c.get('source','');out=[]
  for o in c.get('outputs',[]):
   if o.get('output_type')=='error':errors+=1;out.append('<pre>'+esc(o.get('ename','')+': '+o.get('evalue',''))+'</pre>')
   if 'text' in o:out.append('<pre>'+esc(''.join(o['text']) if isinstance(o['text'],list) else o['text'])+'</pre>')
   data=o.get('data',{})
   if 'image/png' in data:out.append('<img alt="Original notebook output" src="data:image/png;base64,'+''.join(data['image/png'])+'">')
   elif 'text/html' in data:out.append(clean_html(''.join(data['text/html'])))
   elif 'text/plain' in data:out.append('<pre>'+esc(''.join(data['text/plain']))+'</pre>')
  if out:output_cells+=1
  cells.append({'type':c['cell_type'],'source':source,'execution_count':c.get('execution_count'),'output':''.join(out)})
 notebooks.append({'path':path,'title':title,'cells':cells,'errors':errors,'output_cells':output_cells,'code_cells':sum(c['type']=='code' for c in cells),'status':'Executed output archive' if output_cells else 'Source notebook; measured artifacts stored separately'})
# Bundle all selected original files and vector exports. HTML itself is packaged separately.
with zipfile.ZipFile(S/'genai-source-evidence.zip','w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
 for path in files:z.writestr(path,base64.b64decode(files[path]['b64']))
 z.writestr('README.txt','GenAI viva study sources for 23f3000839. Notebook and source bytes are unchanged except the separately documented M4 reviewed interpretation. Model checkpoints and competition data are required to rerun training/inference and are not included. See source-manifest.json for hashes. The study guide labels source archives versus executed outputs.\n')
zip_data=b64((S/'genai-source-evidence.zip').read_bytes())
mean=pd.read_csv(R/'reports/final-results/inference/equal_mean_ensemble-audit-metrics.csv').select_dtypes('number').mean()
payload={'built':stamp,'commit':commit,'concepts':[dict(zip(['id','title','group','meaning','project','viva'],x)) for x in CONCEPTS],'quiz':[dict(zip(['topic','question','answer'],x)) for x in QUIZ],'milestones':MILESTONES,'architectures':shapes,'architectureSVG':svgs,'architectureDetails':details,'pipelineSVG':pipeline_svg,'pipelineDetails':pipeline_details,'splitSVG':split_svg,'dataset':readj('reports/milestone-1-data-results.json'),'modelComparison':pd.read_csv(R/'reports/final-results/model_comparison.csv').to_dict('records'),'variantComparison':pd.read_csv(R/'reports/final-results/inference/inference_comparison.csv').to_dict('records'),'selection':readj('reports/final-results/inference/final_selection.json'),'wandb':readj('study-guide/evidence/wandb-live-verification.json'),'sources':source_entries,'figures':figures,'samples':readj('study-guide/evidence/samples.json'),'notebooks':notebooks,'files':files,'bundle':zip_data,'auditComponents':[float(mean.foreground_dice),float(1-mean.fp_pixel_fraction),float(mean.empty_accuracy)],'manifest':manifest}
serialized=json.dumps(payload,separators=(',',':'),ensure_ascii=False).replace('</','<\\/')
template=(S/'guide-template.html').read_text();result=template.replace('<!--STYLE-->','<style>'+(S/'guide.css').read_text()+'</style>').replace('<!--DATA-->','<script type="application/json" id="project-data">'+serialized+'</script>').replace('<!--SCRIPT-->','<script>'+(S/'guide.js').read_text()+'</script>')
(S/'genai-viva-guide.html').write_text(result)
print('HTML bytes',len(result.encode()),'source bundle bytes',(S/'genai-source-evidence.zip').stat().st_size,'concepts',len(CONCEPTS),'viva questions',len(QUIZ),'notebooks',len(notebooks),'download files',len(files))
