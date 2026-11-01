"""Rebuild the public guide without training data, weights, credentials or network.
Run: .venv-guide/bin/python genai-project/scripts/build_study_site.py
"""
from pathlib import Path
import base64,gzip,json,re,hashlib,html,copy,warnings,io,zipfile
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from pygments import highlight
from pygments.lexers import PythonLexer,JavascriptLexer,TextLexer
from pygments.formatters import HtmlFormatter
from study_lessons import FIGURE_LESSONS,MODEL_LESSONS,NOTEBOOK_GUIDES
from draw_study_architectures import diagram
R=Path(__file__).resolve().parents[1];S=R/'study-guide'
d=json.loads(gzip.decompress((S/'evidence/project-data.json.gz').read_bytes()))
d['figureLessons']=FIGURE_LESSONS;d['modelLessons']=MODEL_LESSONS;d['notebookGuides']=NOTEBOOK_GUIDES
for name in MODEL_LESSONS:
 svg=diagram(name);(S/'diagrams'/f'{name}.svg').write_text(svg);d['architectureSVG'][name]=svg
refs=[{'id':'unet','name':'Original U-Net architecture','file':'unet-original.png','url':'https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/','credit':'Olaf Ronneberger, Philipp Fischer and Thomas Brox · University of Freiburg · 2015','reading':'Start at the upper left. Blue boxes are feature maps. The left arm compresses spatial detail; the right arm restores resolution. Horizontal arrows bring earlier features into the decoder. The numbers on boxes are channels and spatial dimensions.','difference':'This original paper uses cropped skip features, a 572 × 572 example and two output classes. Our custom model uses padded convolutions, bilinear upsampling, BatchNorm, a 128 × 800 grayscale input and four output channels. Read its exact shapes in the project diagram below.'},{'id':'fpn','name':'Original FPN lateral building block','file':'fpn-original.png','url':'https://arxiv.org/html/1612.03144v2#S3','credit':'Tsung-Yi Lin and coauthors · Feature Pyramid Networks for Object Detection · Figure 3 · 2017','reading':'The coarser feature comes down the top-down path and is upsampled. The lateral feature comes from the encoder through a 1×1 convolution. The + combines matching positions by addition. This lets a finer map use deeper context.','difference':'The paper introduces an object-detection pyramid. Our segmentation implementation uses 128 pyramid channels, converts four pyramid scales to 64 channels on one grid, adds them and predicts masks. The original paper’s detection heads are not part of our model.'}]
refs.append({'id':'resnet','name':'Original ResNet residual block','file':'resnet-original.png','url':'https://arxiv.org/abs/1512.03385','credit':'Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun · Figure 2 · 2015','reading':'The input x splits into two routes. Weight layers compute a transformation F(x); the shortcut carries x around them. The + adds the two tensors and ReLU follows. The block learns a correction to the input representation.','difference':'This explains an encoder residual shortcut, not a U-Net encoder-to-decoder skip. Our ResNet34 encoder uses residual stages; its separate U-Net decoder concatenates long skips. When shapes change, a projection shortcut is needed. Our one-channel input and four-mask head are task-specific.'})
for ref in refs:
 raw=(S/'references'/ref['file']).read_bytes();ref['data']='data:image/png;base64,'+base64.b64encode(raw).decode();ref['sha256']=hashlib.sha256(raw).hexdigest()
d['referenceFigures']=refs
fmt=HtmlFormatter(nowrap=True)
for source in d['sources'].values():source['highlighted']=highlight(source['code'],PythonLexer(),fmt)
d['sourceModules']=[]
for path,f in d['files'].items():
 if path.startswith('src/') and path.endswith('.py'):
  code=base64.b64decode(f['b64']).decode();d['sourceModules'].append({'path':path,'sha256':f['sha256'],'lines':len(code.splitlines()),'highlighted':highlight(code,PythonLexer(),fmt),'code':code})
exporter=HTMLExporter(template_name='basic');exporter.exclude_input_prompt=False;exporter.exclude_output_prompt=False
exporter.sanitize_html=False
notebook_verification=[]
for item in d['notebooks']:
 p=R/item['path'];raw=p.read_bytes();original=json.loads(raw)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');nb=nbformat.reads(raw.decode(),as_version=4)
 # Static fallback for progress widgets: show their original text/plain, never start a widget kernel.
 for cell in nb.cells:
  for output in cell.get('outputs',[]):
   output.get('data',{}).pop('application/vnd.jupyter.widget-view+json',None)
 rendered,_=exporter.from_notebook_node(nb)
 soup=BeautifulSoup(rendered,'html.parser')
 for tag in soup(['script','style','iframe','object','embed']):tag.decompose()
 for tag in soup.find_all(True):
  for attr in list(tag.attrs):
   if attr.lower().startswith('on'):del tag[attr]
  for attr in ['href','src']:
   value=str(tag.get(attr,''))
   if value.lower().strip().startswith('javascript:'):del tag[attr]
 for i,cell in enumerate(soup.select('.cell')):
  cell['id']='notebook-cell-'+str(i+1)
  badge=soup.new_tag('div',attrs={'class':'cell-counter'});badge.string='Cell '+str(i+1)+' · '+('Python code' if 'code_cell' in cell.get('class',[]) else 'Notebook notes');cell.insert(0,badge)
  if 'code_cell' in cell.get('class',[]) and not original['cells'][i].get('outputs'):
   note=soup.new_tag('p',attrs={'class':'cell-empty-output'});note.string='No output saved for this cell in the original file.';cell.append(note)
 import bleach
 allowed=['div','span','p','h1','h2','h3','h4','h5','h6','pre','code','strong','em','b','i','u','s','br','hr','ul','ol','li','blockquote','a','img','table','thead','tbody','tfoot','tr','th','td','dl','dt','dd','sup','sub','details','summary']
 item['rendered']=bleach.clean(str(soup),tags=allowed,attributes={'*':['class','id'],'a':['href','title'],'img':['src','alt','width','height'],'th':['colspan','rowspan'],'td':['colspan','rowspan']},protocols=['https','http','data','attachment','mailto'],strip=True)
 item['outline']=[{'cell':i+1,'title':next((line.lstrip('# ').strip() for line in ''.join(c.get('source',[])).splitlines() if line.startswith('#')),'Python cell '+str(i+1))[:95]} for i,c in enumerate(original['cells'])]
 notebook_verification.append({'path':item['path'],'sha256':hashlib.sha256(raw).hexdigest(),'cells':len(original['cells']),'saved_output_cells':item['output_cells'],'rendered_cells':len(soup.select('.cell')),'rendered_images':len(soup.find_all('img'))})
# Extra artifacts remain clearly separate from notebook cell outputs.
d['artifactTexts']={}
for path,f in d['files'].items():
 if path.endswith(('.csv','.json')) and path.startswith('reports/'):
  d['artifactTexts'][path]=base64.b64decode(f['b64']).decode()
# Keep downloadable derived diagrams and their manifest consistent with the displayed SVGs.
for name,svg in d['architectureSVG'].items():
 path='study-guide/diagrams/'+name+'.svg';raw=svg.encode();d['files'][path]={'mime':'image/svg+xml','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'b64':base64.b64encode(raw).decode()}
manifest_path='study-guide/evidence/source-manifest.json'
d['manifest']=[dict(path=path,bytes=f['bytes'],sha256=f['sha256']) for path,f in d['files'].items() if path!=manifest_path]
raw=(json.dumps(d['manifest'],indent=2)+'\n').encode();(S/'evidence/source-manifest.json').write_bytes(raw)
d['files'][manifest_path]={'mime':'application/json','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'b64':base64.b64encode(raw).decode()}
bundle=io.BytesIO()
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
 for path,f in d['files'].items():z.writestr(path,base64.b64decode(f['b64']))
 z.writestr('README.txt','Original project notebooks and results, with checkpoint-derived teaching diagrams. Notebooks retain original saved outputs and original bytes. The new diagrams summarize the actual inspected tensor shapes. Checkpoints and competition data are not included.\n')
(S/'genai-source-evidence.zip').write_bytes(bundle.getvalue());d['bundle']=base64.b64encode(bundle.getvalue()).decode()
css=(S/'guide.css').read_text()+'\n'+(S/'reader.css').read_text()+'\n'+HtmlFormatter().get_style_defs('.python-highlight')
template=(S/'guide-template.html').read_text()
payload=json.dumps(d,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
result=template.replace('<!--STYLE-->','<style>'+css+'</style>').replace('<!--DATA-->','<script type="application/json" id="project-data">'+payload+'</script>').replace('<!--SCRIPT-->','<script>'+(S/'guide.js').read_text()+'</script>')
(S/'genai-viva-guide.html').write_text(result)
(S/'notebook-render-verification.json').write_text(json.dumps({'renderer':'Jupyter nbconvert '+__import__('nbconvert').__version__,'execution':'No cells executed; original saved outputs rendered.','notebooks':notebook_verification},indent=2)+'\n')
print('Built',len(result.encode()),'bytes;',len(d['notebooks']),'Jupyter-rendered notebooks; original bytes preserved.')
