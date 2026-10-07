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
from exile_v4 import smooth_tail
KEYS=[('C',0,0),('Db',1,-5),('D',2,2),('Eb',3,-3),('E',4,4),('F',5,-1),('Gb',6,-6),('G',7,1),('Ab',8,-4),('A',9,3),('Bb',10,-2),('B',11,5)]
SHARP=[('C',0),('C',1),('D',0),('D',1),('E',0),('F',0),('F',1),('G',0),('G',1),('A',0),('A',1),('B',0)]
FLAT=[('C',0),('D',-1),('D',0),('E',-1),('E',0),('F',0),('G',-1),('G',0),('A',-1),('A',0),('B',-1),('B',0)]

def key_signature(es):
 hist=np.zeros(12)
 for n in es:hist[n['pitch']%12]+=min(1.2,n['end']-n['start'])*np.sqrt(max(n.get('level',.01),.0001))
 major=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]);minor=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
 results=[]
 for key,pc,fifths in KEYS:
  coverage=sum(hist[(pc+k)%12] for k in [0,2,4,5,7,9,11])/hist.sum()
  corr=max(np.corrcoef(hist,np.roll(major,pc))[0,1],np.corrcoef(hist,np.roll(minor,(pc+9)%12))[0,1])
  results.append((coverage*.65+corr*.35-.0005*abs(fifths),key,fifths,coverage))
 _,key,fifths,coverage=max(results)
 return key,fifths,float(coverage)

def midi(es,beats,cycles,path,info):
 tpq=960
 def tick(sec):
  if sec<beats[0]:return max(0,round((sec-beats[0])/(beats[1]-beats[0])*tpq))
  return max(0,round(float(np.interp(sec,beats,np.arange(len(beats))))*tpq))
 # Start the first pulse at zero; fractional intro is represented in the first tempo.
 if beats[0]<0:beats=beats.copy();beats[0]=0
 numerator,denominator=map(int,info.get('meter','4/4').split('/'))
 meta=[(0,-2,b'\xff\x58\x04'+bytes([numerator,int(round(math.log2(denominator))),36 if denominator==8 else 24,8])),(0,-1,b'\xff\x59\x02'+bytes([info['fifths']%256,0]))]
 for i in range(len(beats)-1):
  us=int(round((beats[i+1]-beats[i])*1e6));meta.append((i*tpq,0,b'\xff\x51\x03'+us.to_bytes(3,'big')))
 tracks=[meta]
 for hand,ch in [('right',0),('left',1)]:
  title=('Piano '+hand).encode();ev=[(0,-4,b'\xff\x03'+t.varlen(len(title))+title),(0,-3,bytes([0xc0+ch,0])),(0,-2,bytes([0xb0+ch,91,26]))]
  for a,z in cycles:ev.extend([(tick(a),-1,bytes([0xb0+ch,64,85])),(tick(z),-2,bytes([0xb0+ch,64,0]))])
  for n in es:
   if n['hand']!=hand:continue
   a=tick(n['start']);z=max(a+1,tick(n['end']));ev.extend([(a,1,bytes([0x90+ch,n['pitch'],n['velocity']])),(z,0,bytes([0x80+ch,n['pitch'],0]))])
  ev.append((max(tick(max(n['end'] for n in es)+1.5),max(e[0] for e in ev)+1),2,bytes([0xb0+ch,64,0])));tracks.append(ev)
 raw=b'MThd'+struct.pack('>IHHH',6,1,3,tpq)
 for events in tracks:
  last=0;body=b''
  for at,_,msg in sorted(events):body+=t.varlen(at-last)+msg;last=at
  body+=b'\x00\xff\x2f\x00';raw+=b'MTrk'+struct.pack('>I',len(body))+body
 path.write_bytes(raw)
