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

def xml_check(path,notes,bar_ticks=16):
 tree=ET.parse(path);events=[];active={};ids={'melody':'1','inner':'2','accompaniment':'3','bass':'4'}
 for bar,m in enumerate(tree.findall('./part/measure')):
  cursor=0;last_onset=0
  for e in m:
   if e.tag=='backup':cursor-=int(e.findtext('duration'))
   elif e.tag=='note':
    length=int(e.findtext('duration'));chord=e.find('chord') is not None;at=last_onset if chord else cursor
    if not chord:last_onset=at;cursor+=length
    assert 0<=at<bar_ticks and at+length<=bar_ticks
    if e.find('rest') is not None:continue
    p=e.find('pitch');pitch=12*(int(p.findtext('octave'))+1)+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[p.findtext('step')]+int(p.findtext('alter','0'));voice=e.findtext('voice');key=voice,pitch;q=bar*bar_ticks+at
    if any(x.get('type')=='stop' for x in e.findall('tie')):
     assert key in active and active[key]['end_q']==q
     active[key]['end_q']=q+length
    else:
     n=dict(pitch=pitch,voice=voice,q=q,end_q=q+length);events.append(n);active[key]=n
 assert sorted((n['pitch'],ids[n['voice']],n['q'],n['end_q']) for n in notes)==sorted((n['pitch'],n['voice'],n['q'],n['end_q']) for n in events)
