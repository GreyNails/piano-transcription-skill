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

def render(ns,cycles):
 sr=t.SR;raw=t.sample_bank();bank={}
 for p in {n['pitch'] for n in ns}:
  near=min(raw,key=lambda k:abs(k-p));src=raw[near].copy();src-=src.mean();speed=2**((p-near)/12)
  # Polyphase sample-rate conversion avoids the treble interpolation artifacts
  # of the earlier nearest-sample linear resampling path.
  from fractions import Fraction
  ratio=Fraction(1/speed).limit_denominator(600)
  tone=signal.resample_poly(src,ratio.numerator,ratio.denominator).astype(np.float32)
  tone*=.07/max(float(np.sqrt(np.mean(tone[round(.02*sr):round(.22*sr)]**2))),.006)
  bank[p]=tone
 ends=[]
 for n in ns:
  end=n['end']
  for a,z in cycles:
   if a<=end<z:end=z;break
  ends.append(end)
 endtime=max(max(ends),max(n['end'] for n in ns))+4
 mix=np.zeros((round(endtime*sr),2),np.float32);long={}
 for p in bank:
  length=max(end-n['start']+.6 for n,end in zip(ns,ends) if n['pitch']==p)
  long[p]=b.smooth_tail(bank[p],p,round(length*sr)+5,sr)
 nextstrike={};latest={}
 for i in range(len(ns)-1,-1,-1):
  p=ns[i]['pitch'];nextstrike[i]=latest.get(p);latest[p]=ns[i]['start']
 for i,(n,end) in enumerate(zip(ns,ends)):
  release=float(np.interp(n['pitch'],[21,48,72,108],[.34,.25,.17,.13]));count=round((end-n['start']+release)*sr)
  tone=long[n['pitch']][:count].copy();tt=np.arange(len(tone))/sr
  # MIDI velocity affects hammer brightness as well as loudness.
  cutoff=float(np.interp(n['velocity'],[37,55,75,94],[2600,3600,6000,9500]))
  tone=signal.sosfilt(signal.butter(2,cutoff,fs=sr,output='sos'),tone).astype(np.float32)
  tone[:min(len(tone),round(.0015*sr))]*=np.linspace(0,1,min(len(tone),round(.0015*sr)))
  relstart=end-n['start'];releaseenv=np.ones(len(tone));mask=tt>relstart
  releaseenv[mask]=np.exp(-5*(tt[mask]-relstart)/release)*np.maximum(0,1-(tt[mask]-relstart)/release)
  tone*=releaseenv
  if nextstrike[i] is not None:
   offset=nextstrike[i]-n['start'];mask=tt>offset
   tone[mask]*=.45+.55*np.exp(-(tt[mask]-offset)/.045)
  gain=(n['velocity']/90)**1.65
  pan=float(np.clip((n['pitch']-61)/85,-.25,.25));gains=[math.sqrt((1-pan)/2),math.sqrt((1+pan)/2)]
  at=round(n['start']*sr);z=min(len(mix),at+len(tone))
  for ch in range(2):mix[at:z,ch]+=tone[:z-at]*gain*gains[ch]
 dry=mix.copy();rng=np.random.default_rng(61006)
 for ch in range(2):
  size=round(1.45*sr);tt=np.arange(size)/sr
  ir=signal.sosfilt(signal.butter(2,[220,4300],fs=sr,btype='bandpass',output='sos'),rng.standard_normal(size))
  ir*=np.exp(-6.9078*tt/1.2)*np.minimum(1,np.maximum(0,tt-.024)/.04);ir*=.082/max(np.linalg.norm(ir),1e-8)
  for delay,gain in ([(.024,.085),(.046,.043)] if ch==0 else [(.031,.078),(.058,.039)]):ir[round(delay*sr)]+=gain
  mix[:,ch]+=signal.oaconvolve(.82*dry[:,ch]+.18*dry[:,1-ch],ir,mode='full')[:len(mix)]
 mix=signal.sosfilt(signal.butter(2,28,fs=sr,btype='highpass',output='sos'),mix,axis=0)
 mix[-round(.8*sr):]*=np.linspace(1,0,round(.8*sr))[:,None]
 mix*=.88/max(np.max(abs(mix)),1e-8)
 wav=W/'natural_render.wav';wavfile.write(wav,sr,mix.astype(np.float32))
 # Measure once, then apply a single constant gain. No moving loudness gain
 # can turn soft phrase endings up or compress the original dynamic contour.
 probe=subprocess.run([t.FF,'-hide_banner','-i',str(wav),'-af','loudnorm=I=-18:TP=-1.5:LRA=20:print_format=json','-f','null','-'],capture_output=True,text=True,check=True)
 rawjson=probe.stderr[probe.stderr.rfind('{'):];stats=json.JSONDecoder().raw_decode(rawjson)[0]
 gain=min(-18-float(stats['input_i']),-1.8-float(stats['input_tp']))
 subprocess.run([t.FF,'-v','error','-y','-i',str(wav),'-af',f'volume={gain:.6f}dB','-ar','44100','-codec:a','libmp3lame','-b:a','256k',str(OUT/(TITLE+'.mp3'))],check=True)
 return wav,dict(source_integrated_lufs=float(stats['input_i']),constant_gain_db=gain,estimated_output_lufs=float(stats['input_i'])+gain,estimated_true_peak_db=float(stats['input_tp'])+gain,dynamic_loudness_processing=False)
