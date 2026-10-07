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
NODE=os.environ['PIANO_NODE']

def playable_hands(es):
 groups=[]
 for n in sorted(es,key=lambda n:(n['start'],n['pitch'])):
  if groups and n['start']-groups[-1][0]['start']<.070:groups[-1].append(n)
  else:groups.append([n])
 changed=0
 for group in groups:
  if len(group)<2:continue
  ns=sorted(group,key=lambda n:n['pitch']);choices=[]
  for split in range(len(ns)+1):
   left,right=ns[:split],ns[split:]
   cost=0
   for notes,hand in [(left,'left'),(right,'right')]:
    if not notes:continue
    span=notes[-1]['pitch']-notes[0]['pitch']
    cost+=4*max(0,span-12)+4*max(0,len(notes)-5)
    cost+=sum(1.0*(n['hand']!=hand) for n in notes)
    cost+=sum(max(0,n['pitch']-72)*.6 if hand=='left' else max(0,48-n['pitch'])*.6 for n in notes)
   choices.append((cost,split))
  _,split=min(choices)
  for i,n in enumerate(ns):
   hand='left' if i<split else 'right';changed+=n['hand']!=hand;n['hand']=hand
 return changed
