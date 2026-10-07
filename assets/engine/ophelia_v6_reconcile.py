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

def match(a,b,tol=.07):
 result={}
 for p in range(21,109):
  aa=[i for i,n in enumerate(a) if n['pitch']==p];bb=[i for i,n in enumerate(b) if n['pitch']==p]
  if not aa or not bb:continue
  cost=abs(np.array([a[i]['start'] for i in aa])[:,None]-np.array([b[i]['start'] for i in bb])[None,:]);ii,jj=optimize.linear_sum_assignment(np.where(cost<=tol,cost,1000))
  for i,j in zip(ii,jj):
   if cost[i,j]<=tol:result[aa[i]]=bb[j]
 return result
