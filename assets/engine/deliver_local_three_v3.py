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
import batch_deliver as b
import score_optimize as s

def segments(notes,voice,bars,bar_ticks=32,beat_ticks=8):
 out=[];ns=[n for n in notes if n['voice']==voice]
 for bar in range(bars):
  a=bar*bar_ticks;z=a+bar_ticks;active=[n for n in ns if n['q']<z and n['end_q']>a];grid=[]
  for q in range(a,z):grid.append((sorted({n['pitch'] for n in active if n['q']<=q<n['end_q']}),{n['pitch'] for n in active if n['q']==q}))
  segs=[];i=0
  while i<bar_ticks:
   ps=grid[i][0];j=i+1
   while j<bar_ticks and grid[j][0]==ps and not grid[j][1]:j+=1
   tied=[p for p in ps if any(n['pitch']==p and n['q']<a+i<n['end_q'] for n in active)];cursor=i
   while cursor<j:
    allowed=[]
    for dur in [48,32,24,16,12,8,6,4,3,2,1]:
     if dur>j-cursor:continue
     if dur>=8 and cursor%beat_ticks:continue
     if dur in [4,6] and cursor%4:continue
     if dur==3 and cursor%4 not in [0,1]:continue
     if dur==2 and cursor%2:continue
     allowed.append(dur)
    duration=max(allowed or [1]);segs.append(dict(pitches=ps,ticks=duration,tied=ps if cursor>i else tied,staccato=False));cursor+=duration
   i=j
  out.append(segs)
 return out

def score(name,ns,beats,info):
 qn=[];previous={};collisions=[]
 grouped_ticks={}
 notation_pulse=None
 if False:
  attacks=np.array([n['start'] for n in ns]);best=None
  for bpm in np.arange(127.8,128.201,.025):
   ticks=np.round(attacks*bpm/15);fit=least_squares(lambda x:attacks-x[0]-ticks*x[1],[0,15/bpm],loss='soft_l1',f_scale=.012).x
   ticks=np.round((attacks-fit[0])/fit[1]);fit=least_squares(lambda x:attacks-x[0]-ticks*x[1],fit,loss='soft_l1',f_scale=.012).x
   error=float(np.median(abs(attacks-fit[0]-ticks*fit[1])))
   if best is None or error<best[0]:best=(error,fit)
  phase,step=best[1];notation_pulse=dict(bpm=15/step,phase_seconds=phase,median_residual_ms=best[0]*1000)
  for n in ns:grouped_ticks[n['id']]=max(0,int(round((n['start']-phase)/step))*2)
 if False:
  groups=[]
  for n in ns:
   if groups and n['start']-groups[-1][0]['start']<.035:groups[-1].append(n)
   else:groups.append([n])
  for group in groups:
   tick=max(0,int(round(np.interp(np.median([n['start'] for n in group]),beats,np.arange(len(beats)))*8)))
   for n in group:grouped_ticks[n['id']]=tick
 for n in ns:
  q=grouped_ticks.get(n['id'],max(0,int(round(np.interp(n['start'],beats,np.arange(len(beats)))*8))))
  if n['pitch'] in previous and q<=previous[n['pitch']]:collisions.append(dict(id=n['id'],pitch=n['pitch'],start=n['start'],original_tick=q));q=previous[n['pitch']]+1
  end=max(q+1,int(round(np.interp(n['end'],beats,np.arange(len(beats)))*8)))
  if notation_pulse is not None:end=max(q+2,int(round((n['end']-phase)/step))*2)
  qn.append(dict(n,q=q,end_q=end));previous[n['pitch']]=q
 qn.sort(key=lambda n:(n['q'],n['pitch']));previous={}
 for n in qn:
  if n['pitch'] in previous:previous[n['pitch']]['end_q']=min(previous[n['pitch']]['end_q'],n['q'])
  previous[n['pitch']]=n
 voices=copy.deepcopy(qn)
 for n in voices:n['q']/=2;n['end_q']/=2
 s.assign_voices(voices);byid={n['id']:n for n in voices}
 for n in qn:n['hand']=byid[n['id']]['hand'];n['voice']=byid[n['id']]['voice']
 for n in ns:n['hand']=byid[n['id']]['hand'];n['voice']=byid[n['id']]['voice']
 notation_release_edits=[]
 if False:
  # The audio includes pedal resonance. Writing all resonant tails as held
  # keys produces unplayable overlapping chords. In the reading score, each
  # moving voice releases at its next attack; playback keeps measured gates.
  for voice in ['melody','inner','accompaniment','bass']:
   times=sorted({n['q'] for n in qn if n['voice']==voice})
   next_tick={a:z for a,z in zip(times,times[1:])}
   for n in qn:
    if n['voice']!=voice:continue
    maximum=next_tick.get(n['q'],n['end_q'])
    if n['end_q']>maximum:
     notation_release_edits.append(dict(id=n['id'],old_end_q=n['end_q'],new_end_q=maximum));n['end_q']=maximum
 bars=math.ceil(max(n['end_q'] for n in qn)/32);data=dict(title=name+'_定点修复v3',displayTitle=name,bpm=info['bpm'],meter='4/4',meterEstimated=True,key=info['key'],fifths=info['fifths'],pitchSpellings=copy.deepcopy(b.SHARP if info['fifths']>0 else b.FLAT),ticksPerQuarter=8,staves=[['melody','inner'],['accompaniment','bass']],rehearsalMarks={str(i):str(i//8+1) for i in range(0,bars,8)},performanceText='Cantabile. Playback follows measured source timing; accompany lightly.',reconstructed=True,barsPerSystem=1,suppressRepeatedNoteSlurs=True,footerText='Revision from supplied local media. No external model. Previous editions retained.',engravingInstruction='Independent voices; preserve each repeated strike. Pedal with changing harmony.')
 if False:data.update(systemsPerPage=3,staffGap=180,notationReleaseEdits=notation_release_edits,notationSustain='Moving voices release at the next written attack; acoustic resonance is carried by the pedal.')
 if notation_pulse is not None:data.update(notationPulseFit=notation_pulse,bpm=round(notation_pulse['bpm'],2))
 data['voiceMeasures']={v:segments(qn,v,bars) for v in ['melody','inner','accompaniment','bass']};hands=[segments([dict(n,voice=n['hand']) for n in qn],hand,bars) for hand in ['right','left']];data['measures']=[[hands[0][i],hands[1][i]] for i in range(bars)];return data,qn,collisions
