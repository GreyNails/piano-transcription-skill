"""Portable subset of the verified local piano pipeline."""
import os,sys,json,copy,csv,hashlib,math,struct,subprocess,shutil,time
from pathlib import Path
import numpy as np
from scipy import signal,ndimage,optimize
from scipy.io import wavfile
from scipy.optimize import least_squares,linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
import xml.etree.ElementTree as ET
import transcribe as t
W=t.WORK/'visual';BAR_ROWS=[94,144,194];KEYBOARD_ROW=380;STRIKE_ROW=328

def log(name,*args):print(name,*args,flush=True)

def stage(name):
 p=W/name;p.mkdir(exist_ok=True);return p

def rows(name):
 cache=stage(name)/'rows.npz'
 if cache.exists():return np.load(cache)['rgb']
 ys=BAR_ROWS+[KEYBOARD_ROW]
 filters='[0:v]fps=30,scale=1088:612,split=4[a][b][c][d];'+''.join(f'[{c}]crop=1088:6:0:{y},scale=1088:1[{c}o];' for c,y in zip('abcd',ys))+'[ao][bo][co][do]vstack=inputs=4[out]'
 proc=subprocess.Popen([t.FF,'-v','error','-threads','2','-i',str(t.WORK/'videos'/f'{name}.mp4'),'-filter_complex_threads','1','-filter_complex',filters,'-map','[out]','-an','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
 frames=[];size=1088*4*3
 while True:
  buf=proc.stdout.read(size)
  if len(buf)!=size:break
  frames.append(np.frombuffer(buf,np.uint8).reshape(4,1088,3))
 if proc.wait():raise RuntimeError('decode failed')
 rgb=np.stack(frames);np.savez_compressed(cache,rgb=rgb);log(name,'rows',rgb.shape)
 return rgb

def pitch_features(name,y):
 cache=stage(name)/'pitch.npz'
 if cache.exists():return dict(np.load(cache))
 # Pitch-specific attacks at 5 ms; raw evidence is also used for velocity.
 hop=110;nfft=4096
 _,tt,z=signal.stft(y,t.SR,nperseg=nfft,noverlap=nfft-hop,boundary='zeros');mag=np.abs(z).astype(np.float32)
 E=[];A=[]
 flux=np.maximum(mag-np.pad(mag[:,:-5],((0,0),(5,0))),0)
 for p in range(21,109):
  f=440*2**((p-69)/12);ee=0;aa=0
  for h in range(1,9):
   k=round(f*h*nfft/t.SR)
   if k>=len(mag)-2:break
   ee+=mag[max(0,k-1):k+2].max(0)/h**1.2
   aa+=flux[max(0,k-1):k+2].max(0)/h**.7
  E.append(ee);A.append(aa)
 E=np.array(E);A=np.array(A)
 F=A/np.maximum(np.percentile(A,99,axis=1,keepdims=True),1e-6)
 F=np.minimum(F,3)
 np.savez_compressed(cache,E=E,F=F,tt=tt)
 return dict(E=E,F=F,tt=tt)

def visual_events(name,y):
 rgb=rows(name);bright=rgb[:,:3].max(3).astype(np.float32)
 xs=t.keys(-3,21.05)
 # Find the dark-key centers on the actual keyboard, then infer white-key spacing.
 keyboard=np.percentile(rgb[:,3].mean(2),35,axis=0)
 dark,_=signal.find_peaks(-ndimage.gaussian_filter1d(keyboard,1),distance=12,prominence=12)
 for k,p in enumerate(range(21,109)):
  if p%12 in [1,3,6,8,10] and len(dark):
   nearest=dark[np.argmin(abs(dark-xs[k]))]
   if abs(nearest-xs[k])<9:xs[k]=nearest
 # Refine centers from clean falling-bar components, retaining the key identity.
 centers=[[] for _ in range(88)]
 a=bright[:,0]
 mask=(a>75)&(a-ndimage.minimum_filter1d(a,size=31,axis=1)>35)
 labs,_=ndimage.label(mask)
 for i,s in enumerate(ndimage.find_objects(labs)):
  f,x=s
  if not (1<=f.stop-f.start and 5<=x.stop-x.start<=24):continue
  if (labs[s]==i+1).mean()<.60:continue
  center=(x.start+x.stop-1)/2;k=int(np.argmin(abs(xs-center)))
  if abs(xs[k]-center)<8:centers[k].append(center)
 for k in range(88):
  if len(centers[k])>=5:xs[k]=np.median(centers[k])
 curves=[]
 for r in range(3):
  b=bright[:,r];background=ndimage.minimum_filter1d(b,size=31,axis=1)
  b=np.maximum(0,b-background)
  curves.append(np.array([np.median(b[:,max(0,round(x)-2):min(1088,round(x)+3)],axis=1) for x in xs]))
 # Track movement down the screen. The same bar has to occur on independent rows.
 correlations=[]
 aa=np.maximum(curves[0]-50,0);bb=np.maximum(curves[1]-50,0)
 for lag in range(2,121):
  a=aa[:,:-lag];b=bb[:,lag:];correlations.append(float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)))
 lag=2+int(np.argmax(correlations));speed=(BAR_ROWS[1]-BAR_ROWS[0])*30/lag
 row_es=[]
 for r in range(3):
  b=bright[:,r];local=b-ndimage.minimum_filter1d(b,size=31,axis=1)
  saturation=rgb[:,r].max(2).astype(float)-rgb[:,r].min(2).astype(float)
  mask=(b>70)&(local>40)&(saturation>18)
  mask=ndimage.binary_opening(mask,structure=np.ones((1,3)))
  labs,_=ndimage.label(mask)
  ev=[]
  for lab,sl in enumerate(ndimage.find_objects(labs)):
   f,x=sl;frames=f.stop-f.start;width=x.stop-x.start
   if not (1<=frames<=1200 and 5<=width<=29):continue
   component=labs[sl]==lab+1
   if component.mean()<.50:continue
   # Assign each bar ONCE by its center, rather than sampling overlapping keys.
   center=(x.start+x.stop-1)/2;k=int(np.argmin(abs(xs-center)))
   if abs(xs[k]-center)>7:continue
   delay=(STRIKE_ROW-BAR_ROWS[r])/speed
   ev.append(dict(pitch=k+21,start=f.start/30+delay,end=f.stop/30+delay,row=r,strength=float(b[sl].max()),frames=int(frames)))
  row_es.append(ev)
 raw=[]
 for p in range(21,109):
  groups=[[n for n in ev if n['pitch']==p] for ev in row_es]
  # Union of supported trajectories, so one weak/occluded row cannot erase notes.
  candidates=sorted(sum(groups,[]),key=lambda n:n['start'])
  used=set()
  for n in candidates:
   if id(n) in used:continue
   matches=[min(g,key=lambda x:abs(x['start']-n['start'])) for g in groups if g]
   matches=[m for m in matches if abs(m['start']-n['start'])<.080]
   matches=list({id(m):m for m in matches}.values())
   if len({m['row'] for m in matches})<2:continue
   if max(m['frames'] for m in matches)<2:continue
   used.update(id(m) for m in matches)
   raw.append(dict(pitch=p,start=float(np.median([m['start'] for m in matches])),end=float(np.median([m['end'] for m in matches])),visual_rows=len(matches)))
 raw.sort(key=lambda n:(n['start'],n['pitch']))
 # These Jova keyboards span all 88 keys, A0–C8. Preserve that absolute map;
 # normalized harmonic flux alone can favor an incorrect lower octave.
 feat=pitch_features(name,y);F=feat['F'];E=feat['E'];tt=feat['tt'];hop=tt[1]-tt[0]
 subset=raw[::max(1,len(raw)//1200)]
 starts=np.array([n['start'] for n in subset]);pitches=np.array([n['pitch'] for n in subset])-21
 best=(-1,0,0)
 for transpose in [0]:
  pp=pitches+transpose;valid=(pp>=0)&(pp<88)
  for offset in np.arange(-1.0,1.501,.015):
   ix=np.clip(np.round((starts[valid]+offset+.015)/hop).astype(int),0,F.shape[1]-1)
   val=float(np.mean(F[pp[valid],ix]))
   if val>best[0]:best=(val,transpose,float(offset))
 _,transpose,offset=best
 # Offset can drift with the mux timebase. Fit robust local window offsets.
 windows=[]
 for center in np.arange(15,len(y)/t.SR,25):
  ns=[n for n in raw if abs(n['start']-center)<18 and 21<=n['pitch']+transpose<=108]
  if len(ns)<15:continue
  ps=np.array([n['pitch']+transpose-21 for n in ns]);ss=np.array([n['start'] for n in ns])
  offsets=np.arange(offset-.18,offset+.181,.005)
  vals=[np.mean(F[ps,np.clip(np.round((ss+o+.015)/hop).astype(int),0,F.shape[1]-1)]) for o in offsets]
  windows.append([center,float(offsets[np.argmax(vals)]),float(max(vals))])
 if len(windows)>3:
  ww=np.array(windows);fit=least_squares(lambda a:(ww[:,0]*a[0]+a[1]-ww[:,1]),[0,offset],loss='soft_l1',f_scale=.025).x
 else:fit=np.array([0,offset])
 es=[]
 for n in raw:
  p=n['pitch']+transpose
  if not 21<=p<=108:continue
  pred=n['start']*(1+fit[0])+fit[1]
  if pred<0 or pred>len(y)/t.SR-.1:continue
  a=max(0,np.searchsorted(tt,pred-.055));b=min(len(tt),np.searchsorted(tt,pred+.07))
  if a>=b:continue
  k=a+np.argmax(F[p-21,a:b]);strength=float(F[p-21,k])
  start=float(tt[k]-.020) if strength>.12 else pred
  end=max(start+.035,n['end']*(1+fit[0])+fit[1])
  level=float(np.percentile(E[p-21,max(0,k-1):min(len(tt),k+18)],80))
  es.append(dict(pitch=p,start=max(0,start),end=min(end,len(y)/t.SR),hand='left' if p<60 else 'right',attack=strength,level=level,visual_rows=n['visual_rows']))
 info=dict(method='three-row luminous-bar tracking with audio pitch and attack alignment',raw_notes=len(raw),motion_row_correlation=max(correlations),pixels_per_second=speed,octave_shift=transpose,affine_time_fit=fit.tolist(),alignment_windows=windows)
 (stage(name)/'visual_debug.json').write_text(json.dumps(dict(xs=xs.tolist(),info=info),indent=2))
 log(name,'VISUAL',len(es),'transpose',transpose,'offset/drift',fit.tolist(),'corr',round(max(correlations),3))
 return es,info
