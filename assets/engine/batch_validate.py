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

def vl(raw,pos):
 n=0
 while True:
  val=raw[pos];pos+=1;n=(n<<7)|(val&127)
  if val<128:return n,pos

def read_midi(path):
 raw=path.read_bytes();headlen,kind,ntracks,tpq=struct.unpack('>IHHH',raw[4:14]);pos=8+headlen;ev=[];tempos=[];pedal={};active={};notes=[]
 for track in range(ntracks):
  assert raw[pos:pos+4]==b'MTrk';length=int.from_bytes(raw[pos+4:pos+8],'big');end=pos+8+length;pos+=8;at=0;status=None
  while pos<end:
   delta,pos=vl(raw,pos);at+=delta
   if raw[pos]>=128:status=raw[pos];pos+=1
   assert status is not None
   if status==255:
    typ=raw[pos];pos+=1;length,pos=vl(raw,pos);data=raw[pos:pos+length];pos+=length
    if typ==81:tempos.append((at,int.from_bytes(data,'big')))
   elif status in [240,247]:length,pos=vl(raw,pos);pos+=length
   else:
    ch=status&15;typ=status>>4;length=1 if typ in [12,13] else 2;data=raw[pos:pos+length];pos+=length
    if typ==9 and data[1]:
     k=ch,data[0];assert k not in active,f'Overlapping MIDI gate {k}'
     active[k]=(at,data[1])
    elif typ==8 or typ==9 and not data[1]:
     k=ch,data[0];assert k in active,f'Unmatched note off {k}';start,vel=active.pop(k);assert at>start
     notes.append(dict(pitch=data[0],hand='right' if ch==0 else 'left',start_tick=start,end_tick=at,velocity=vel))
    elif typ==11 and data[0]==64:pedal[ch]=data[1]
 assert not active,'Stuck MIDI notes';assert all(v==0 for v in pedal.values()),'Pedal left down'
 tempos.sort();ts=[];seconds=[];us=500000;last=0;now=0
 for at,tempo in tempos:
  now+=(at-last)/tpq*us/1e6;ts.append(at);seconds.append(now);last=at;us=tempo
 def time(tick):
  i=max(0,int(np.searchsorted(ts,tick,side='right'))-1)
  return seconds[i]+(tick-ts[i])/tpq*tempos[i][1]/1e6
 for n in notes:n['start']=time(n['start_tick']);n['end']=time(n['end_tick'])
 return notes
