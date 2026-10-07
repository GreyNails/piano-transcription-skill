"""Constrained phrase-level velocity calibration against the supplied recording."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d,median_filter
import transcribe as t
import yanhua_natural as synth
W=t.WORK/'yanhua_v4';ORIGIN=.35;NAME='烟花易冷';SOURCE_AUDIO=None;MASK_INTERVALS=[]

def main():
 notes=json.loads((W/'performance_uncalibrated.json').read_text());pedal=json.loads((W/'pedal.json').read_text());raw=W/'uncalibrated';raw.mkdir(exist_ok=True)
 synth.W=raw;synth.OUT=raw;synth.TITLE='uncalibrated'
 signature=hashlib.sha256(json.dumps([notes,pedal],sort_keys=True).encode()).hexdigest()
 signature_path=raw/'input_signature.txt'
 if not (raw/'natural_render.wav').exists() or not signature_path.exists() or signature_path.read_text()!=signature:
  synth.render(notes,pedal);signature_path.write_text(signature)
 sr,source=wavfile.read(SOURCE_AUDIO or t.WORK/'audio'/f'{NAME}.wav');source=source.astype(np.float64)/32768;source=source[round(ORIGIN*sr):]
 rr,render=wavfile.read(raw/'natural_render.wav');assert sr==rr
 hop=round(sr*.01)
 def env(x):
  if x.ndim==2:x=x.mean(1)
  return np.sqrt(np.mean(np.pad(x,(0,(-len(x))%hop)).reshape(-1,hop)**2,axis=1)+1e-12)
 es=env(source);er=env(render);n=min(len(es),len(er));es=es[:n];er=er[:n]
 # A broad contour adjusts phrase dynamics, never individual subdivisions.
 ss=gaussian_filter1d(es,55);rs=gaussian_filter1d(er,55);floor=max(np.percentile(ss,90)*.03,1e-6)
 active=(ss>floor)&(rs>np.percentile(rs,90)*.025)
 ratio=20*np.log10((ss+floor*.1)/(rs+np.percentile(rs,90)*.003+1e-8))
 valid=np.ones(n,dtype=bool);source_times=np.arange(n)*hop/sr+ORIGIN
 for a,z in MASK_INTERVALS:valid&=~((source_times>=a-.6)&(source_times<=z+.6))
 if not valid.all():ratio[~valid]=np.interp(source_times[~valid],source_times[valid],ratio[valid])
 center=float(np.median(ratio[active&valid]));delta=gaussian_filter1d(median_filter(ratio-center,size=101),60);delta=np.clip(delta,-4,4)
 # Do not boost silence or extend phrase endings; note inventory and gates
 # are fixed. Preserve within-chord velocity differences.
 at=np.arange(n)*hop/sr;changes=[]
 for x in notes:
  db=float(np.interp(x['start'],at,delta));old=x['velocity'];v=int(np.clip(round(old*10**(db/(20*1.65))),max(1,old-12),min(115,old+12)))
  x['velocity']=v
  if old!=v:changes.append(dict(id=x['id'],source_time=x['start']+ORIGIN,pitch=x['pitch'],old_velocity=old,new_velocity=v,phrase_gain_db=db))
 (W/'performance.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2));(W/'dynamics_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2));(W/'dynamics_report.json').write_text(json.dumps(dict(method='source phrase RMS / same-sample renderer RMS, 550ms smoothing; 1s median; 600ms final smoothing',max_contour_db=4,max_velocity_change=12,within_chord_ranking_not_explicitly_changed=True,note_timing_changed=False,pedal_changed=False,adjusted_notes=len(changes),constant_gain_offset_removed_db=center),indent=2))
 np.savez_compressed(W/'dynamics_contour.npz',tt=at,gain_db=delta,source=ss,uncalibrated=rs)
 print('calibrated',len(changes),'notes; timing and releases unchanged',flush=True)
if __name__=='__main__':main()
