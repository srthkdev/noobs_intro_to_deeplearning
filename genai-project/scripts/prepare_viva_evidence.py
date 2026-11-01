"""Inspect the actual trained checkpoints and replay three audit examples on CPU."""
from pathlib import Path
import sys,json,hashlib,time,base64,io
import numpy as np,pandas as pd
from PIL import Image
import torch,segmentation_models_pytorch as smp
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from final_pipeline import make_model,decode_rle,encode_rle
from predict_final import Segmenter
OUT=R/'study-guide/evidence';OUT.mkdir(parents=True,exist_ok=True)
TRAIN=R.parent/'final-training/results';DATA=R.parent/'data/public'
torch.set_num_threads(4)
records=[]
def shapes(x):
 if isinstance(x,torch.Tensor):return list(x.shape)
 if isinstance(x,(list,tuple)):return [shapes(v) for v in x]
 return str(type(x))
comparison=pd.read_csv(R/'reports/final-results/model_comparison.csv').set_index('model')
for name in comparison.index:
 path=TRAIN/name/'best.pt';saved=torch.load(path,map_location='cpu',weights_only=True);model=make_model(name,pretrained=False).eval();model.load_state_dict(saved['state_dict']);n=sum(p.numel() for p in model.parameters());assert n==int(comparison.loc[name,'parameters'])
 stages={};handles=[]
 def hook(label):
  def record(module,inputs,output):stages[label]={'module':module.__class__.__name__,'input':shapes(inputs),'output':shapes(output)}
  return record
 if name=='scratch_unet':
  for i,b in enumerate(model.enc):handles.append(b.register_forward_hook(hook(f'enc.{i}')))
  for i,b in enumerate(model.dec):handles.append(b.register_forward_hook(hook(f'dec.{i}')))
  handles.append(model.head.register_forward_hook(hook('head')))
 else:
  handles.append(model.encoder.register_forward_hook(hook('encoder')))
  for nm,module in model.decoder.named_children():
   handles.append(module.register_forward_hook(hook('decoder.'+nm)))
   if nm=='blocks':
    for i,b in enumerate(module):handles.append(b.register_forward_hook(hook('decoder.blocks.'+str(i))))
  handles.append(model.decoder.register_forward_hook(hook('decoder')))
  handles.append(model.segmentation_head.register_forward_hook(hook('segmentation_head')))
 with torch.inference_mode():output=model(torch.zeros(1,1,128,800))
 assert list(output.shape)==[1,4,128,800]
 for h in handles:h.remove()
 (OUT/(name+'-architecture.txt')).write_text(str(model)+'\n')
 info={'name':name,'parameters':n,'checkpoint_epoch':saved['epoch'],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checkpoint_config':saved['config'],'stages':stages,'output_shape':list(output.shape),'inspection':'Loaded actual saved weights; CPU forward pass verifies stage tensor shapes. No training or new selection.','inspection_runtime':{'torch':torch.__version__,'smp':smp.__version__},'architecture_text':name+'-architecture.txt'}
 records.append(info);print(name,n,'stages',json.dumps(stages),flush=True)
 del model
(OUT/'architectures.json').write_text(json.dumps(records,indent=2)+'\n')
# Use two good-overlap audit examples and one recorded false positive, with frozen selection.
pairs=pd.read_csv(R/'reports/final-results/inference/audit-pair-metrics.csv');ids=[]
for cls in ['defect_c','defect_b']:
 rows=pairs[(pairs.defect_type==cls)&(pairs.target_pixels>0)&(pairs.dice>.75)].sort_values('image_id')
 image_id=str(rows.iloc[0].image_id);ids.append((image_id,'Good overlap example: '+cls))
ids.append(('RKRFKJ0HOQ','Recorded false positive on an empty defect_c target'))
raw=pd.read_csv(DATA/'train.csv',keep_default_na=False);table=raw.pivot(index='image_id',columns='defect_type',values='mask_rle').reindex(columns=['defect_a','defect_b','defect_c','defect_d'])
segmenter=Segmenter(TRAIN,R/'reports/final-results/inference/final_selection.json',device='cpu')
samples=[]
for image_id,why in ids:
 t=time.perf_counter();pred,probs=segmenter.predict_batch([DATA/'train'/(image_id+'.jpg')]);elapsed=time.perf_counter()-t
 with Image.open(DATA/'train'/(image_id+'.jpg')) as f:
  b=io.BytesIO();f.convert('L').save(b,format='PNG');img='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
 truth=[str(x).strip() for x in table.loc[image_id]];metrics=[]
 for c,label in enumerate(table.columns):
  gt=decode_rle(truth[c])>0;pr=pred[0,c];pa=int(pr.sum());ga=int(gt.sum());inter=int((pr&gt).sum());dice=2*inter/(pa+ga) if pa+ga else 1
  stored=pairs[(pairs.image_id==image_id)&(pairs.defect_type==label)].iloc[0]
  metrics.append({'class':label,'target_pixels':ga,'prediction_pixels':pa,'dice':dice,'recorded_kaggle_prediction_pixels':int(stored.prediction_pixels),'recorded_kaggle_dice':float(stored.dice)})
 sample={'image_id':image_id,'purpose':why,'partition':'audit','image':img,'truth_rle':truth,'prediction_rle':[encode_rle(pred[0,c]) for c in range(4)],'probabilities_f16_le':base64.b64encode(probs[0].astype('<f2').tobytes()).decode(),'metrics':metrics,'cpu_replay_seconds':elapsed,'source':'Frozen final checkpoints replayed on CPU for this study guide. Original audit metrics retained alongside replay values.'}
 samples.append(sample);print('replayed',image_id,round(elapsed,2),[(v['class'],v['prediction_pixels'],v['recorded_kaggle_prediction_pixels']) for v in metrics],flush=True)
(OUT/'samples.json').write_text(json.dumps(samples,separators=(',',':')))
print('Saved three genuine audit examples and complete checkpoint architecture records.',flush=True)
