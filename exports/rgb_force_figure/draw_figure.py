from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-context-figure')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
out=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(13,13))
fig.subplots_adjust(left=.025,right=.975,top=.975,bottom=.025)
ax.set(xlim=(0,14),ylim=(0,15));ax.axis('off')
C={'rgb':'#F6DCD5','ft':'#D7EAF3','attn':'#F9E2B9','norm':'#F3F0C8','out':'#D8EBD8','query':'#E4DDF0','ink':'#343C46'}
def box(x,y,w,h,label,color='white',fs=11,lw=1.4):
 p=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.035,rounding_size=0.07',fc=color,ec=C['ink'],lw=lw,zorder=3)
 ax.add_patch(p);ax.text(x,y,label,ha='center',va='center',fontsize=fs,color=C['ink'],zorder=4);return p
def arrow(x,y,X,Y):
 ax.add_patch(FancyArrowPatch((x,y),(X,Y),arrowstyle='-|>',mutation_scale=12,lw=1.35,color=C['ink'],zorder=2))
def line(points):
 ax.plot(*zip(*points),color=C['ink'],lw=1.25,zorder=2)
def plus(x,y):
 ax.add_patch(Circle((x,y),.12,fc='white',ec=C['ink'],lw=1.3,zorder=4));ax.text(x,y,'+',ha='center',va='center',fontsize=13,zorder=5)
def note(x,y,s,fs=10,ha='left'):
 ax.text(x,y,s,ha=ha,va='center',fontsize=fs,color='#5D6670',linespacing=1.5)
ax.text(7,14.68,'RGB–Force Context Classifier',ha='center',fontsize=20,weight='bold',color=C['ink'])
ax.text(7,14.28,'Frozen visual backbone · Causal force histories · Transformer fusion',ha='center',fontsize=11,color='#5D6670')
# Inputs and feature encoders: bottom to top.
for x,title,shape,col in [(2.35,'RGB observations','2 × 3 × 224 × 224',C['rgb']),(7,'Left F/T history','41 × 6',C['ft']),(11.65,'Right F/T history','41 × 6',C['ft'])]:
 box(x,1.02,3.8,.75,title+'\n'+shape,col)
 arrow(x,1.43,x,1.84)
box(2.35,2.25,3.8,.75,'Pretrained CLIP ViT-B/16\nFrozen · CLIP normalization',C['rgb'])
box(2.35,3.38,3.8,.75,'Linear projection\n768 → 128 (per image)',C['rgb'])
arrow(2.35,2.66,2.35,2.97)
for x,side in [(7,'Left'),(11.65,'Right')]:
 box(x,2.25,3.8,.75,'Raw + mean + mean difference\n41 × 6 → 32 × 18',C['ft'],10.5)
 box(x,3.38,3.8,.75,'Feature normalization\n'+side+' causal CNN → 128',C['ft'])
 arrow(x,2.66,x,2.97)
for x,label in [(2.35,'2 image tokens'),(7,'1 left-force token'),(11.65,'1 right-force token')]:
 arrow(x,3.79,x,4.16)
 note(x,4.34,label,10.5,'center')
 line([(x,4.54),(x,4.8),(7,4.8)])
arrow(7,4.8,7,5.08)
box(7,5.46,7.15,.68,'Concatenate: [RGB t−3, RGB t, F/T left, F/T right, query]','white',10.4)
box(1.64,5.46,2.5,.68,'Learnable query',C['query'],10.5)
arrow(2.93,5.46,3.37,5.46)
arrow(7,5.83,7,6.12)
box(7,6.48,5.6,.64,'+ Learned positional embeddings','white')
note(10.25,6.48,'5 tokens × 128 dimensions',10)
note(.52,6.4,'Modality embeddings added\nto image / left / right features',9.5)
# Transformer detailed pre-norm layer.
container=FancyBboxPatch((4.12,7.03),5.76,4.2,boxstyle='round,pad=0.04,rounding_size=0.16',fc='#F5F6F7',ec=C['ink'],lw=1.6,zorder=0)
ax.add_patch(container)
arrow(7,6.84,7,7.29)
box(7,7.54,3.5,.42,'Layer Norm',C['norm'])
arrow(7,7.78,7,7.99)
box(7,8.37,3.5,.68,'Masked Multi-Head\nSelf-Attention',C['attn'])
arrow(7,8.75,7,9.04);plus(7,9.17)
line([(7,7.14),(4.6,7.14),(4.6,9.17)]);arrow(4.6,9.17,6.85,9.17)
arrow(7,9.3,7,9.48)
box(7,9.72,3.5,.42,'Layer Norm',C['norm'])
arrow(7,9.96,7,10.12)
box(7,10.4,3.5,.48,'Feed Forward  128 → 256 → 128',C['ft'],9.3)
arrow(7,10.68,7,10.91);plus(7,11.04)
line([(7,9.38),(9.42,9.38),(9.42,11.04)]);arrow(9.42,11.04,7.15,11.04)
note(10.25,9.55,'2× encoder layers\n4 attention heads\nPre-norm · residual additions\nDropout = 0.1',10.5)
note(.52,8.65,'Attention combines the tokens.\nThe final query can attend to\nall four observation tokens\nand itself.',10)
# Readout.
arrow(7,11.2,7,11.57)
box(7,11.86,4.35,.48,'Select final query representation',C['query'],10.5)
arrow(7,12.13,7,12.35)
box(7,12.59,3.5,.42,'Linear 128 → 4',C['query'])
arrow(7,12.83,7,13.03)
box(7,13.27,3.5,.42,'Softmax',C['out'])
arrow(7,13.51,7,13.72)
ax.text(7,13.92,'Approach  /  Turning  /  Recovery  /  Error',ha='center',va='center',fontsize=11,weight='bold',color=C['ink'])
note(.52,12.65,'OUTPUT\nFour state probabilities',10.5)
note(10.25,12.55,'TRAINING\nWeighted cross-entropy\nManual / rule labels: 1.0 / 0.3\nInverse-frequency class weights',10)
ax.text(7,.37,'Shapes omit batch dimension. Only CLIP is frozen; projection, CNNs, embeddings, query and Transformer are trained.',ha='center',fontsize=9,color='#5D6670')
ax.text(7,.1,'F/T: force and torque. Mean window = 5; mean-difference lag = 5. Residual-path dropout omitted for clarity.',ha='center',fontsize=9,color='#5D6670')
for ext in ['svg','pdf','png']:
 fig.savefig(out/f'rgb_force_context_architecture.{ext}',dpi=240,facecolor='white')
print(out)
