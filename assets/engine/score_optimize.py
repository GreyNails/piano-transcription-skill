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
from batch_finalize import NODE,playable_hands

def decode(data):
 events=[];active=[{},{}];bar_ticks=data.get('barTicks',4*data.get('ticksPerQuarter',4))
 for bar,measure in enumerate(data['measures']):
  for hand,segs in enumerate(measure):
   q=bar*bar_ticks
   for seg in segs:
    for p in seg['pitches']:
     if p in seg.get('tied',[]) and p in active[hand] and active[hand][p]['end_q']==q:
      active[hand][p]['end_q']=q+seg['ticks']
     else:
      n=dict(pitch=p,q=q,end_q=q+seg['ticks'],hand='right' if hand==0 else 'left',source_bar=bar+1,source_tick=q%bar_ticks)
      n['id']=len(events);events.append(n);active[hand][p]=n
    q+=seg['ticks']
 return sorted(events,key=lambda n:(n['q'],n['pitch']))

def assign_voices(notes):
 changes=playable_hands(notes);groups={}
 for n in notes:groups.setdefault((n['q'],n['hand']),[]).append(n)
 rh=[n for n in notes if n['hand']=='right'];rq=np.array([n['q'] for n in rh]);rp=np.array([n['pitch'] for n in rh]);suspects=[]
 for (q,hand),ns in groups.items():
  ns.sort(key=lambda n:n['pitch'])
  if hand=='right':
   local=rp[abs(rq-q)<32];median=float(np.median(local)) if len(local) else ns[-1]['pitch']
   candidates=[n for n in ns if n['pitch']<=median+19] or ns
   lead=candidates[-1]
   for n in ns:
    n['voice']='melody' if n is lead else 'inner'
    if n['pitch']>median+19 and n['end_q']-n['q']<=4:
     suspects.append(dict(bar=q//16+1,beat=round(q%16/4+1,2),pitch=n['pitch'],reason='孤立的高音区跳跃；保留音高，需核对原音频'))
  else:
   low=ns[0]
   is_bass=low['pitch']<50 or (low['pitch']<=58 and (q%4==0 or len(ns)>1))
   for n in ns:n['voice']='bass' if n is low and is_bass else 'accompaniment'
 # Mark chromatic notes only when rare AND isolated; no scale snapping or deletion.
 pcs=np.bincount([n['pitch']%12 for n in notes],minlength=12)
 for n in notes:
  if pcs[n['pitch']%12]<max(3,len(notes)*.003) and n['end_q']-n['q']<=2:
   suspects.append(dict(bar=n['q']//16+1,beat=round(n['q']%16/4+1,2),pitch=n['pitch'],reason='少见的短时变化音；保留音高，可能是经过音、转调或转录疑点'))
 return changes,suspects

def xml_score(data,path):
 unit=data.get('ticksPerQuarter',4);bar_ticks=data.get('barTicks',4*unit);meter=data.get('meter','4/4').split('/')
 root=ET.Element('score-partwise',version='4.0');work=ET.SubElement(root,'work');ET.SubElement(work,'work-title').text=data['displayTitle']+' - edited piano score'
 identification=ET.SubElement(root,'identification');ET.SubElement(identification,'creator',type='arranger').text='Editorial version of the supplied transcription'
 pl=ET.SubElement(root,'part-list');sp=ET.SubElement(pl,'score-part',id='P1');ET.SubElement(sp,'part-name').text='Piano';part=ET.SubElement(root,'part',id='P1')
 types={1:'16th',2:'eighth',3:'eighth',4:'quarter',6:'quarter',8:'half',12:'half',16:'whole'}
 if unit==8:types={1:'32nd',2:'16th',3:'16th',4:'eighth',6:'eighth',8:'quarter',12:'quarter',16:'half',24:'half',32:'whole',48:'whole'}
 voiceids={'melody':1,'inner':2,'accompaniment':3,'bass':4};voices=['melody','inner','accompaniment','bass'];bars=len(data['measures'])
 for bar in range(bars):
  m=ET.SubElement(part,'measure',number=str(bar+1))
  if bar==0:
   a=ET.SubElement(m,'attributes');ET.SubElement(a,'divisions').text=str(unit);ET.SubElement(ET.SubElement(a,'key'),'fifths').text=str(data['fifths']);time=ET.SubElement(a,'time');ET.SubElement(time,'beats').text=meter[0];ET.SubElement(time,'beat-type').text=meter[1];ET.SubElement(a,'staves').text='2'
   for staff,sign,line in [(1,'G',2),(2,'F',4)]:
    c=ET.SubElement(a,'clef',number=str(staff));ET.SubElement(c,'sign').text=sign;ET.SubElement(c,'line').text=str(line)
   direction=ET.SubElement(m,'direction',placement='above');ET.SubElement(ET.SubElement(direction,'direction-type'),'words').text=data['performanceText'];ET.SubElement(direction,'sound',tempo=str(data['bpm']))
  if str(bar) in data['rehearsalMarks']:
   direction=ET.SubElement(m,'direction',placement='above');ET.SubElement(ET.SubElement(direction,'direction-type'),'rehearsal').text=data['rehearsalMarks'][str(bar)]
  if bar==0:
   direction=ET.SubElement(m,'direction',placement='below');ET.SubElement(ET.SubElement(ET.SubElement(direction,'direction-type'),'dynamics'),'mp');ET.SubElement(direction,'staff').text='1'
  active=[]
  for staff in data['staves']:
   present=[v for v in staff if any(s['pitches'] for s in data['voiceMeasures'][v][bar])];active.extend(present or [staff[0]])
  for vi,voice in enumerate(active):
   if vi:ET.SubElement(ET.SubElement(m,'backup'),'duration').text=str(bar_ticks)
   staff=1 if voice in ['melody','inner'] else 2;segs=data['voiceMeasures'][voice][bar]
   for j,seg in enumerate(segs):
    nxt=segs[j+1] if j+1<len(segs) else (data['voiceMeasures'][voice][bar+1][0] if bar+1<bars else None)
    for index,p in enumerate(seg['pitches'] or [None]):
     n=ET.SubElement(m,'note')
     if index:ET.SubElement(n,'chord')
     if p is None:ET.SubElement(n,'rest')
     else:
      letter,alter=data['pitchSpellings'][p%12];pc={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[letter]+alter
      pit=ET.SubElement(n,'pitch');ET.SubElement(pit,'step').text=letter
      if alter:ET.SubElement(pit,'alter').text=str(alter)
      ET.SubElement(pit,'octave').text=str((p-pc)//12-1)
     ET.SubElement(n,'duration').text=str(seg['ticks']);ties=[]
     if p is not None and p in seg['tied']:ties.append('stop')
     if p is not None and nxt and p in nxt['tied']:ties.append('start')
     for typ in ties:ET.SubElement(n,'tie',type=typ)
     ET.SubElement(n,'voice').text=str(voiceids[voice]);ET.SubElement(n,'type').text=types[seg['ticks']]
     if seg['ticks'] in ([3,6,12,24,48] if unit==8 else [3,6,12]):ET.SubElement(n,'dot')
     if p is not None:ET.SubElement(n,'stem').text='up' if voice in ['melody','accompaniment'] else 'down'
     ET.SubElement(n,'staff').text=str(staff)
     if ties:
      notation=ET.SubElement(n,'notations')
      for typ in ties:ET.SubElement(notation,'tied',type=typ)
  if bar==bars-1:ET.SubElement(ET.SubElement(m,'barline',location='right'),'bar-style').text='light-heavy'
 ET.indent(root);ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
