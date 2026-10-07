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

def command(args):return subprocess.run(args,check=True,capture_output=True,text=True)

def measure(p):
 r=command([t.FF,'-hide_banner','-i',str(p),'-af','loudnorm=I=-18:TP=-1.5:LRA=20:print_format=json','-f','null','-'])
 return json.JSONDecoder().raw_decode(r.stderr[r.stderr.rfind('{'):])[0]
