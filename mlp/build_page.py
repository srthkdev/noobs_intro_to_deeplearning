"""Build one continuous study page from the unchanged executed final notebook.
Run from repo root: .venv-guide/bin/python mlp/build_page.py
No notebook cells are executed.
"""
from pathlib import Path
import base64,csv,hashlib,html,json,re,warnings
import nbformat,bleach
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from content import STEPS,GLOSSARY,VIVA,OUTPUT_NOTES,PARAMETERS
R=Path(__file__).resolve().parent
esc=lambda s:html.escape(str(s),quote=True)
read=lambda p:json.loads((R/p).read_text())
notebook_path=R/'project/notebooks/final.ipynb';raw=notebook_path.read_bytes();n=json.loads(raw)
summary=read('project/reports/final-results/summary.json');submission=read('project/reports/final-results/kaggle-submission.json');conf=read('evidence/audit-confusion.json')
models=list(csv.DictReader((R/'project/reports/final-results/model_comparison.csv').open()))
assert len(n['cells'])==15 and sum(bool(c.get('outputs')) for c in n['cells'])==7
assert sum(o.get('output_type')=='error' for c in n['cells'] for o in c.get('outputs',[]))==0
assert abs(conf['macro_f1']-summary['selected_audit_macro_f1'])<1e-12
exporter=HTMLExporter(template_name='basic');exporter.sanitize_html=False
with warnings.catch_warnings():
 warnings.simplefilter('ignore');nb=nbformat.reads(raw.decode(),as_version=4);rendered,_=exporter.from_notebook_node(nb)
soup=BeautifulSoup(rendered,'html.parser')
for el in soup(['script','style','iframe','object','embed']):el.decompose()
allowed=['div','span','p','h1','h2','h3','h4','h5','h6','pre','code','strong','em','b','i','br','hr','ul','ol','li','blockquote','a','img','table','thead','tbody','tfoot','tr','th','td','sup','sub','details','summary']
attrs={'*':['class','id'],'a':['href','title'],'img':['src','alt','width','height'],'th':['colspan','rowspan'],'td':['colspan','rowspan']}
cells=soup.select('.cell');assert len(cells)==15
line_stops={};rendered_cells={};output_check=[]
for i,(cell,original) in enumerate(zip(cells,n['cells']),1):
 cell['id']=f'nb-cell-{i}'
 if original['cell_type']=='code':
  source=''.join(original['source']);area=cell.select_one('.input_area');assert area
  area.clear();code_soup=BeautifulSoup(highlight(source,PythonLexer(),HtmlFormatter(linenos='inline',linespans=f'cell-{i}-line',anchorlinenos=True)),'html.parser');area.append(code_soup)
  line_stops[i]=[]
  for note in next(s['stops'] for s in STEPS if s['cell']==i):
   needle,title,explanation=note
   matches=[j+1 for j,line in enumerate(source.splitlines()) if needle in line]
   assert matches, (i,needle)
   line_stops[i].append(dict(line=matches[0],title=title,explanation=explanation))
  original_outputs=original.get('outputs',[]);areas=cell.select('.output_area');notes=OUTPUT_NOTES[i]
  assert len(areas)==len(original_outputs)==len(notes),(i,len(areas),len(original_outputs),len(notes))
  for j,(area,(title,explanation)) in enumerate(zip(areas,notes),1):
   wrapper=soup.new_tag('div',attrs={'class':'annotated-output','id':f'cell-{i}-output-{j}'})
   area.wrap(wrapper)
   caption=soup.new_tag('div',attrs={'class':'output-reading'})
   heading=soup.new_tag('h4');heading.string=title;caption.append(heading)
   p=soup.new_tag('p');p.string=explanation;caption.append(p);wrapper.insert(0,caption)
  output_check.append(dict(cell=i,original_outputs=len(original_outputs),rendered_outputs=len(areas),source_sha256=hashlib.sha256(source.encode()).hexdigest()))
 rendered_cells[i]=bleach.clean(str(cell),tags=allowed,attributes=attrs,protocols=['http','https','mailto','data'],strip=True)
# Labels and explanation are outside the original code/outputs.
sections=[]
for j,step in enumerate(STEPS,1):
 i=step['cell'];concepts=''.join(f'<div><strong>{esc(term)}</strong><p>{esc(text)}</p></div>' for term,text in step['terms'])
 stops=''.join(f'<button class="code-stop" onclick="goLine({i},{s["line"]})"><span>Line {s["line"]} · {esc(s["title"])}</span>{esc(s["explanation"])}</button>' for s in line_stops[i])
 extras=''
 if step['id']=='split':extras='<!--SPLIT-->'
 if step['id']=='features':extras='<!--REPRESENTATIONS-->'
 if step['id']=='models':extras='<!--MODELS-->'
 if step['id']=='audit':extras='<!--CONFUSION-->'
 if step['id']=='refit':extras='<!--FINAL_RESULT-->'
 sections.append(f'''<section class="notebook-step" id="{step['id']}" aria-labelledby="title-{step['id']}"><div class="step-heading"><span class="step-number">{j:02d}</span><div><span class="eyebrow">Final notebook · original cell {i}</span><h2 id="title-{step['id']}">{esc(step['title'])}</h2></div></div><div class="reasoning"><div><h3>What we did</h3><p>{esc(step['what'])}</p></div><div><h3>Why</h3><p>{esc(step['why'])}</p></div><div><h3>How it works</h3><p>{esc(step['how'])}</p></div></div><div class="concept-strip">{concepts}</div>{extras}<div class="notebook-original-notes"><span class="source-label">Original notebook notes · cell {i-1}</span>{rendered_cells[i-1]}</div><div class="code-guide"><h3>Where to look in this cell</h3><p>Click a note to jump to the exact original Python line. The highlight is a reading aid.</p><div class="code-stops">{stops}</div></div><div class="cell-toolbar"><strong>Original Python + saved outputs · cell {i}</strong><div><button onclick="expandCode({i},this)">Expand full code</button><button onclick="document.getElementById('cell-{i}-output-1').scrollIntoView({{block:'start',behavior:'smooth'}})">Jump to outputs ↓</button></div></div><div class="notebook-render">{rendered_cells[i]}</div><div class="takeaway"><strong>What to take from the result</strong><p>{esc(step['look'])}</p><h3>Viva: {esc(step['viva'][0])}</h3><p>{esc(step['viva'][1])}</p></div>{f'<div class="notebook-original-notes"><span class="source-label">Original closing notes · cell 15</span>{rendered_cells[15]}</div>' if j==7 else ''}</section>''')
def table(headers,rows):return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def svg_start(width,height,label):return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}"><defs><marker id="{label}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#63868d"/></marker></defs>'
def diagram_box(x,y,w,title,sub,fill='#edf4fa',href=''):
 out=f'<g><rect x="{x}" y="{y}" width="{w}" height="76" rx="12" fill="{fill}" stroke="#c7d6e2"/><text x="{x+15}" y="{y+29}" font-size="16" font-weight="bold" fill="#223f53">{esc(title)}</text><text x="{x+15}" y="{y+55}" font-size="13" fill="#567081">{esc(sub)}</text></g>'
 return f'<a href="#{href}">{out}</a>' if href else out
svg=svg_start(1140,270,'workflow')
for j,(x,y,title,sub,href) in enumerate([(20,25,'1 · Load + EDA','Inspect rows, labels, missingness','eda'),(305,25,'2 · Grouped split','Fit / tune / audit','split'),(590,25,'3 · Preprocess','Fit rules on fitting data','features'),(875,25,'4 · Compare models','Tune selects the winner','models'),(875,165,'5 · Audit','Measure the frozen decision','audit'),(590,165,'6 · Full-data refit','Fixed recipe, all labels','refit'),(305,165,'7 · Submission','Preserve IDs and valid tiers','refit')]):svg+=diagram_box(x,y,245,title,sub,href=href)
for d in ['M265 63H300','M550 63H585','M835 63H870','M997 101V160','M875 203H840','M590 203H555']:
 svg+=f'<path d="{d}" fill="none" stroke="#63868d" stroke-width="2" marker-end="url(#workflow-arrow)"/>'
svg+='</svg>'
split=svg_start(1140,210,'split')
for x,w,label,count,color in [(20,750,'Fit · learn parameters','195,842 rows','#d6e7f9'),(770,110,'Tune','27,978','#d3e9e0'),(880,240,'Audit','55,926','#f9e1c9')]:
 split+=f'<rect x="{x}" y="35" width="{w}" height="56" fill="{color}"/><text x="{x+12}" y="69" fill="#234356" font-size="15">{esc(label)}</text><text x="{x+12}" y="118" font-size="16" fill="#234356">{esc(count)}</text>'
split+='<text x="20" y="155" fill="#567081" font-size="14">Same predictor profile → same partition. Sizes are approximate 70 / 10 / 20; widths are schematic.</text><text x="20" y="182" fill="#567081" font-size="14">Only fit learns preprocessing. Tune chooses settings. Audit reports the frozen choice.</text></svg>'
representation=svg_start(1140,360,'representation')+diagram_box(20,137,245,'Raw listing','Numbers, categories, dates, lists')+diagram_box(340,137,250,'Fit-only feature builder','227 engineered fit columns')
for y,title,sub in [(15,'SGD + Extra Trees','Impute → scale → one-hot · 1,165 cols'),(137,'LightGBM','Numeric + fitted category levels'),(259,'CatBoost','Numeric sentinel + category strings')]:
 representation+=diagram_box(710,y,405,title,sub,'#e3f0eb')
for d in ['M265 175H335','M590 175H650V53H705','M590 175H705','M590 175H650V297H705']:
 representation+=f'<path d="{d}" fill="none" stroke="#63868d" stroke-width="2" marker-end="url(#representation-arrow)"/>'
representation+='</svg>'
model_rows=[]
for r in models:
 model_rows.append([esc(r['model']),*[f'{float(r[k]):.6f}' for k in ['train_macro_f1','tune_macro_f1','audit_macro_f1']],f'{float(r["fit_seconds"])/60:.2f} min'])
model_table=table(['Configuration','Train F1','Tune F1 · selection','Audit F1 · report','Recorded fit time'],model_rows)
parameter_table=table(['Setting','Recorded value','Meaning'],[[esc(v) for v in row] for row in PARAMETERS])
model_extras=f'''<div class="study-insert"><h3>The actual comparison, before reading the training log</h3><p>Five configurations, four families. Higher F1 is better. Compare tuning values to understand the choice; audit values are reported afterwards. These are holdout scores, not CV means.</p>{model_table}<div id="comparison-chart" class="vector-diagram"></div><p class="caption">New plot of the original model_comparison.csv values. Horizontal scale starts at 0; the training gap is visible. No new models were fitted.</p><h3>What the selected LightGBM settings mean</h3>{parameter_table}<div class="contrast"><div><h4>Extra Trees: many independently randomized trees</h4><p>Fit randomized trees, then combine their class predictions. Our configuration has 80 estimators, depth ≤20 and minimum leaf support 5. bootstrap stays False in this run.</p></div><div><h4>Boosting: sequential additions</h4><p>Each stage improves the current loss using gradient information. F<sub>m</sub>(x) = F<sub>m−1</sub>(x) + ηh<sub>m</sub>(x) describes the additive idea. This formula is a teaching schematic, not a dump of learned tree thresholds.</p></div></div></div>'''
confusion='''<div class="study-insert" id="matrix-explorer"><h3>Read the actual audit errors interactively</h3><p>Rows are true tiers; columns are predicted tiers. Click a square to explain that specific count. Switch to row percentages to see the same normalization as the original plot. These counts come from the saved 55,926 audit predictions.</p><div class="row"><label for="matrix-mode">Display</label><select id="matrix-mode"><option value="count">Original counts</option><option value="percent">Percent within true tier</option></select><label for="metric-class">Explain F1 for</label><select id="metric-class"><option value="0">Tier 1</option><option value="1">Tier 2</option><option value="2" selected>Tier 3</option><option value="3">Tier 4</option></select></div><div class="matrix-layout"><div id="matrix"></div><div><div id="matrix-reading" class="reading-card"></div><div id="class-metrics" class="reading-card"></div></div></div><div id="macro-explanation" class="formula"></div></div>'''
final_result=f'''<div class="study-insert"><h3>Three scores; three different evaluation sources</h3>{table(['Source','Recorded result','What it evaluates'],[['Tuning','0.735466 macro F1','The fit-stage model; used for model and stopping selection.'],['Audit','0.738081 macro F1','That frozen fit-stage model on held-out audit labels.'],['Kaggle public','0.70158','Code submission 57777975 from registered notebook version 3.']])}<p>After full-data refitting, the previous audit rows are now training rows. The old audit score does not become an independent score of the refitted object. The hidden-test result is recorded separately.</p><p class="source">Kaggle record: 29 October 2026 · script version 363546821 · status COMPLETE · <a href="https://www.kaggle.com/code/sarthakiitmbs/23f3000839-notebook-2026t3?scriptVersionId=363546821" target="_blank" rel="noreferrer">registered notebook ↗</a></p></div>'''
body=''.join(sections).replace('<!--SPLIT-->','<div class="study-insert"><h3>How information is allowed to flow</h3><div class="vector-diagram">'+split+'</div><p>Initial EDA used the full labelled table. The disjointness boundary here concerns fitted preprocessing, training, early stopping and winner selection; it does not mean no one ever saw audit rows in the exploratory plots.</p></div>').replace('<!--REPRESENTATIONS-->','<div class="study-insert"><h3>One feature builder, three input paths</h3><div class="vector-diagram">'+representation+'</div><p>Read left to right. The shared builder creates meaningful columns; the branches prepare different representations for each model family. The final LightGBM uses its native categorical branch, not the baseline 1,165-column one-hot matrix.</p></div>').replace('<!--MODELS-->',model_extras).replace('<!--CONFUSION-->',confusion).replace('<!--FINAL_RESULT-->',final_result)
glossary=''.join(f'<article class="concept-card" data-filter="{esc(term+" "+meaning)}"><h3>{esc(term)}</h3><p>{esc(meaning)}</p><a href="#{where}">Find it in the final notebook ↑</a></article>' for term,meaning,where in GLOSSARY)
viva=''.join(f'<article class="viva-card" data-filter="{esc(q+" "+a)}"><span class="eyebrow">Question {i:02d} · <a href="#{where}">related cell ↑</a></span><h3>{esc(q)}</h3><p class="viva-answer">{esc(a)}</p></article>' for i,(q,a,where) in enumerate(VIVA,1))
nav=''.join(f'<a href="#{s["id"]}"><span>{i}</span>{esc(s["id"].title())}</a>' for i,s in enumerate(STEPS,1))+'<a href="#concepts">Concepts</a><a href="#viva">Viva</a>'
manifest=[]
for p in sorted((R/'project').rglob('*')):
 if p.is_file():manifest.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(R/'evidence/source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
payload=dict(models=models,confusion=conf,summary=summary,notebook_sha256=hashlib.sha256(raw).hexdigest(),lineStops=line_stops)
payload_json=json.dumps(payload,separators=(',',':')).replace('</','<\\/')
replacements={'<!--NAV-->':nav,'<!--WORKFLOW-->':svg,'<!--ORIGINAL_INTRO-->':rendered_cells[1],'<!--WALKTHROUGH-->':body,'<!--GLOSSARY-->':glossary,'<!--VIVA-->':viva,'<!--CSS-->':'<style>'+(R/'style.css').read_text()+'\n'+HtmlFormatter().get_style_defs('.highlight')+'</style>','<!--DATA-->':'<script id="project-data" type="application/json">'+payload_json+'</script>','<!--JS-->':'<script>'+(R/'script.js').read_text()+'</script>','<!--NBHASH-->':hashlib.sha256(raw).hexdigest()}
page=(R/'template.html').read_text()
for key,val in replacements.items():page=page.replace(key,val)
assert not re.search(r'<!--(?:NAV|CSS|JS|DATA|WALKTHROUGH|GLOSSARY|VIVA)-->',page)
(R/'index.html').write_text(page)
(R/'evidence/render-verification.json').write_text(json.dumps(dict(renderer='Jupyter nbconvert '+__import__('nbconvert').__version__,notebook_sha256=hashlib.sha256(raw).hexdigest(),original_cells=15,original_code_cells=7,original_output_items=sum(x['original_outputs'] for x in output_check),executed_for_guide=False,code_cells=output_check,glossary_entries=len(GLOSSARY),viva_questions=len(VIVA)),indent=2)+'\n')
print('Built one page:',len(page.encode()),'bytes; 15 original cells; 7 annotated code cells;',len(GLOSSARY),'concepts;',len(VIVA),'viva questions.')
