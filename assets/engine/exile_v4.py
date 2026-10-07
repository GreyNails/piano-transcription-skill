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

def smooth_tail(source,pitch,length,sr):
    if length<=len(source):return source[:length].copy()
    # Fit stable partials near the sample's end; a long note does not loop a noisy
    # fragment or abruptly change phase at a fixed two-second sample boundary.
    nfit=min(len(source),round(.4*sr));chunk=source[-nfit:]
    nfft=32768;spec=np.abs(np.fft.rfft(chunk*np.hanning(nfit),nfft));freq=np.fft.rfftfreq(nfft,1/sr)
    peaks,_=signal.find_peaks(spec,distance=max(3,round(15*nfft/sr)))
    peaks=[k for k in peaks if 35<freq[k]<7500]
    peaks=sorted(peaks,key=lambda k:spec[k],reverse=True)[:18]
    frequencies=[]
    for k in peaks:
        a,b,c=np.log(np.maximum(spec[k-1:k+2],1e-10));delta=.5*(a-c)/(a-2*b+c)
        frequencies.append((k+delta)*sr/nfft)
    tt=np.arange(nfit)/sr
    columns=[]
    for f in frequencies:columns.extend([np.sin(2*np.pi*f*tt),np.cos(2*np.pi*f*tt)])
    A=np.stack(columns,axis=1);coef=np.linalg.lstsq(A,chunk,rcond=1e-6)[0]
    cross=round(.12*sr);begin=len(source)-cross
    future=(np.arange(begin,length)-(len(source)-nfit))/sr
    tail=np.zeros(len(future))
    tau=2.5 if pitch<48 else (1.8 if pitch<65 else 1.1)
    for j,f in enumerate(frequencies):
        decay=np.exp(-np.maximum(0,future-nfit/sr)/(tau/(1+f/5500)))
        tail+=(coef[2*j]*np.sin(2*np.pi*f*future)+coef[2*j+1]*np.cos(2*np.pi*f*future))*decay
    out=np.zeros(length,np.float32);out[:len(source)]=source
    fade=np.linspace(0,1,cross)
    out[begin:len(source)]=out[begin:len(source)]*(1-fade)+tail[:cross]*fade
    out[len(source):]=tail[cross:]
    return out
