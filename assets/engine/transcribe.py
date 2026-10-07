"""Portable subset of the verified local piano pipeline."""
import os,sys,json,copy,csv,hashlib,math,struct,subprocess,shutil,time
from pathlib import Path
import numpy as np
from scipy import signal,ndimage,optimize
from scipy.io import wavfile
from scipy.optimize import least_squares,linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
import xml.etree.ElementTree as ET
ROOT=Path(os.environ['PIANO_RUNTIME'])
WORK=ROOT/'work';RESULT=Path(os.environ['PIANO_RESULT'])
FF=os.environ['PIANO_FFMPEG'];SR=22050;FPS=30

def keys(x0=0,w=1088/52,first=21,last=108):
    xx=[];wi=0
    for p in range(first,last+1):
        if p%12 in [1,3,6,8,10]:x=x0+wi*w
        else:x=x0+(wi+.5)*w;wi+=1
        xx.append(x)
    return np.array(xx)

def sample_bank():
    out={}
    for f in (ROOT/'tools/samples').glob('*.wav'):
        sr,y=wavfile.read(f);out[int(f.stem)]=y.astype(np.float32)/32768
    return out

def varlen(n):
    n=max(0,int(n));b=[n&127];n>>=7
    while n:b.insert(0,(n&127)|128);n>>=7
    return bytes(b)
