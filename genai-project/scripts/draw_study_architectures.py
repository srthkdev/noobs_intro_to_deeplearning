"""Draw readable, numbered feature-map diagrams from inspected stage shapes."""
from pathlib import Path
from html import escape
import json
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'study-guide/evidence';D=ROOT/'study-guide/diagrams'
COL={'encoder':('#dae9ff','#547cad'),'decoder':('#d7eee6','#387f6b'),'output':('#fce8d3','#ad6b39'),'input':('#ecedf0','#737d8c')}
def diagram(name):
 W=1320;H=900 if name=='resnet34_unet' else 810
 out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" class="actual-architecture" role="img" aria-label="{name} verified feature map architecture"><defs>']
 for k,c in [('flow','#46766b'),('skip','#c07633')]:out.append(f'<marker id="{name}-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="{c}"/></marker>')
 out.append('</defs><style>.actual-architecture text{font-family:Arial,sans-serif;fill:#26394c}.actual-architecture .small{font-size:14px;fill:#51657b}.actual-architecture .shape{font:17px ui-monospace,monospace;font-weight:bold}.actual-architecture .heading{font-size:19px;font-weight:bold}.layer:focus{outline:none}.layer:focus .frame{stroke:#b05b22;stroke-width:3}.actual-architecture .arch-wire{stroke:#46766b;stroke-width:2.5;fill:none}.actual-architecture .arch-skip{stroke:#c07633;stroke-width:2.5;stroke-dasharray:7 5;fill:none}</style><rect width="100%" height="100%" rx="18" fill="#fafcff"/>')
 def txt(x,y,s,cls='small',anchor='start'):out.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>')
 def node(key,x,y,title,shape,kind='encoder',num=None,sub=''):
  fill,stroke=COL[kind];out.append(f'<g class="layer" data-stage="{key}" tabindex="0" role="button" aria-label="{escape(title+", "+shape)}"><rect class="frame" x="{x-12}" y="{y-12}" width="304" height="96" rx="14" fill="{fill}" fill-opacity=".45" stroke="{stroke}" stroke-opacity=".35"/>')
  for off in [12,6,0]:out.append(f'<rect x="{x+off}" y="{y-off/2}" width="40" height="45" rx="3" fill="{fill}" stroke="{stroke}"/>')
  txt(x+66,y+15,title,'heading');txt(x+66,y+40,shape,'shape');txt(x+66,y+63,sub)
  if num is not None:out.append(f'<circle cx="{x-6}" cy="{y-10}" r="13" fill="{stroke}"/><text x="{x-6}" y="{y-5}" text-anchor="middle" style="font-size:13px;fill:white;font-weight:bold">{num}</text>')
  out.append('</g>')
 def line(d,label='',x=0,y=0,skip=False):
  typ='skip' if skip else 'flow';out.append(f'<path class="{"arch-skip" if skip else "arch-wire"}" d="{d}" marker-end="url(#{name}-{typ})"/>')
  if label:
   width=len(label)*7.3+18;out.append(f'<rect x="{x-width/2}" y="{y-16}" width="{width}" height="23" rx="6" fill="#fafcff"/>');txt(x,y,label,anchor='middle')
 txt(45,38,'BLUE: encoder features','heading');txt(470,38,'GREEN: decoder features','heading');txt(900,38,'ORANGE: class output','heading')
 # Shape notation is explained in the surrounding lesson, leaving the input card clear.
 if name in ('scratch_unet','resnet34_unet'):
  scratch=name=='scratch_unet'; ys=[180,325,470,615] if scratch else [170,300,430,560,690];x=90;dx=800
  enc=['24 × 128 × 800','48 × 64 × 400','96 × 32 × 200','192 × 16 × 100'] if scratch else ['64 × 64 × 400','64 × 32 × 200','128 × 16 × 100','256 × 8 × 50','512 × 4 × 25']
  for i,(y,shape) in enumerate(zip(ys,enc)):
   node('enc'+str(i),x,y,('Block ' if scratch else 'Encoder ')+str(i+1),shape,num=i+2 if scratch else {0:2,1:3,3:4,4:5}.get(i),sub='2 × (Conv → BN → ReLU)' if scratch else ('stem, stride 2' if i==0 else 'ResNet residual stage'))
   if i<len(ys)-1:line(f'M{x+140} {y+84} V{ys[i+1]-14}', 'pool 2×2' if scratch else 'downsample',x+140,(y+84+ys[i+1]-14)/2+4)
  # Input and class output live above opposite branches.
  node('input',x,65,'Input','1 × 128 × 800','input',num=1,sub='one normalized strip')
  # Input card is compact to avoid overlapping the first stage.
  # It is drawn separately above the first feature; route from its right edge to the first stage.
  line(f'M{x+140} 149 V{ys[0]-14}')
  if scratch:
   dec=[(470,'96 × 32 × 200','dec0',6,'288 → 96 channels'),(325,'48 × 64 × 400','dec1',None,'144 → 48 channels'),(180,'24 × 128 × 800','dec2',7,'72 → 24 channels')]
  else:
   dec=[(560,'256 × 8 × 50','dec0',6,'upsample + concat + conv'),(430,'128 × 16 × 100','dec1',None,'upsample + concat + conv'),(300,'64 × 32 × 200','dec2',None,'upsample + concat + conv'),(170,'32 × 64 × 400','dec3',None,'upsample + concat + conv')]
  for j,(y,shape,key,num,sub) in enumerate(dec):
   node(key,dx,y,'Decoder '+str(j+1),shape,'decoder',num,sub)
   encoder_index=len(enc)-2-j;sy=ys[encoder_index]
   line(f'M{x+294} {sy+27} H{dx-14}', 'concatenate channels',580,sy+25,True)
   if j<len(dec)-1:line(f'M{dx+144} {y-13} V{dec[j+1][0]+85}', 'upsample ×2',dx+144,(y-13+dec[j+1][0]+85)/2+4)
  bottom=ys[-1];target=dec[0][0];line(f'M{x+294} {bottom+28} H660 V{target+65} H{dx-14}','upsample ×2',650,bottom+12)
  if not scratch:
   # Final stage is separate above the five-scale U, no extra encoder skip.
   node('dec4',dx,65,'Decoder 5','16 × 128 × 800','decoder',7,'last stage: no encoder skip')
   line(f'M{dx+144} {170-13} V{65+85}')
   node('head',430,65,'Class head','4 × 128 × 800','output',8,'logits → sigmoid outside model')
   line(f'M{dx-14} 94 H724')
  else:
   node('head',dx,65,'1×1 class head','4 × 128 × 800','output',8,'logits → sigmoid outside model');line(f'M{dx+144} 167 V150')
  txt(55,H-33,'Solid green: sequential computation. Dashed orange: encoder skip, joined by channel concatenation.')
 else:
  ys=[175,315,455,595];shapes=['24 × 32 × 200','40 × 16 × 100','112 × 8 × 50','320 × 4 × 25'];p=['128 × 32 × 200','128 × 16 × 100','128 × 8 × 50','128 × 4 × 25']
  node('input',45,65,'Input','1 × 128 × 800','input',1,'EfficientNet-B0 encoder below');line('M185 149 V162')
  for i,y in enumerate(ys):
   node('enc'+str(i),45,y,'Encoder scale '+str(i+2),shapes[i],num=2 if i==0 else None,sub='pretrained features, fine-tuned')
   node('p'+str(i+2),470,y,'P'+str(i+2),p[i],'decoder',{0:5,2:4,3:3}.get(i),'1×1 projection' if i==3 else 'lateral + upsampled deeper')
   line(f'M339 {y+27} H456','1×1',395,y+25,True)
   if i<3:line(f'M185 {y+84} V{ys[i+1]-13}')
   if i<3:line(f'M614 {ys[i+1]-13} V{y+85}','upsample',615,(ys[i+1]-13+y+85)/2+4)
  node('align',925,175,'Align 4 scales','4 × (64 × 32 × 200)','decoder',6,'segmentation blocks + upsample')
  for i,y in enumerate(ys):line(f'M764 {y+28} H{810+i*21} V{203+i*11} H911')
  node('merge',925,380,'Add + dropout','64 × 32 × 200','decoder',7,'addition keeps 64 channels');line('M1069 259 V366','element-wise sum',1069,330)
  node('head',925,595,'Head + 4× resize','4 × 128 × 800','output',8,'logits → sigmoid outside model');line('M1069 464 V581','class head; upsample ×4',1069,531)
  txt(55,H-33,'FPN lateral links use addition after channel projection. All four aligned outputs are also added, not concatenated.')
 out.append('</svg>');return ''.join(out)
if __name__=='__main__':
 for name in ['scratch_unet','resnet34_unet','efficientnet_b0_fpn']:
  (D/(name+'.svg')).write_text(diagram(name))
