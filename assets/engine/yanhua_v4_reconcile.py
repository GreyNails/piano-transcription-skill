"""Source-specific attack evidence for a reversible piano revision."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import json,copy
from pathlib import Path
import numpy as np
from scipy import signal,optimize
import transcribe as t
from ophelia_v6_reconcile import match
NAME='烟花易冷';W=t.WORK/'yanhua_v4';MODELS=t.RESULT/NAME/'开源模型测试_20261006';ORIGIN=.35
OLD_PERFORMANCE=t.WORK/'targeted_v3/烟花易冷/performance.json'
OLD_ORIGIN=None
SOURCE_AUDIO=None

def load(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 a=load(MODELS/'transkun/model_notes.json');b=load(MODELS/'bytedance/model_notes.json');ab=match(a,b,.065)
 candidates=[dict(n,transkun_id=i,bytedance_id=ab.get(i),model_agreement=i in ab) for i,n in enumerate(a)]
 candidates += [dict(n,transkun_id=None,bytedance_id=j,model_agreement=False) for j,n in enumerate(b) if j not in set(ab.values())]
 candidates.sort(key=lambda n:(n['start'],n['pitch']))
 if SOURCE_AUDIO is None:y=t.audio_file(NAME)
 else:
  from scipy.io import wavfile
  sr,y=wavfile.read(SOURCE_AUDIO);assert sr==t.SR;y=y.astype(np.float32)/32768
 size=4096;hop=110;f,tt,z=signal.stft(y,t.SR,nperseg=size,noverlap=size-hop,boundary='zeros');V=abs(z).astype(np.float64);del z
 weight=1/np.maximum(V.mean(1),V.mean(1).max()*.02)**.4;weight[(f<35)|(f>9000)]=0
 def innovation(at):
  bef=(tt>=max(0,at-.085))&(tt<max(.005,at-.040));aft=(tt>=at+.015)&(tt<at+.065)
  before=V[:,bef].mean(1) if bef.any() else np.zeros(len(f));after=V[:,aft].mean(1) if aft.any() else np.zeros(len(f))
  return np.maximum(after-.9*before,0)
 # Each harmonic profile comes from this recording, using model-consensus
 # attacks without a lower simultaneously struck octave/harmonic carrier.
 pitches=sorted({n['pitch'] for n in candidates});D=[];anchors={}
 Y=[innovation(n['start']) for n in candidates]
 for p in pitches:
  f0=440*2**((p-69)/12);mask=np.zeros(len(f));lines=[]
  for h in range(1,32):
   freq=f0*h
   if freq>9000:break
   x=(f-freq)*size/t.SR;line=abs(.5*np.sinc(x)+.25*np.sinc(x-1)+.25*np.sinc(x+1));lines.append(line);mask=np.maximum(mask,line)
  chosen=[]
  for i,n in enumerate(candidates):
   if n['pitch']!=p or not n['model_agreement']:continue
   other=[m for m in candidates if abs(m['start']-n['start'])<.06 and p-m['pitch'] in [12,19,24,28,31,36]]
   if not other:chosen.append(i)
  fallback=False
  if not chosen:chosen=[i for i,n in enumerate(candidates) if n['pitch']==p];fallback=True
  chosen=sorted(chosen,key=lambda i:np.linalg.norm(Y[i]*mask),reverse=True)[:24]
  cols=np.array([Y[i]/max(np.linalg.norm(Y[i]*mask),1e-12) for i in chosen]);profile=np.median(cols,axis=0)
  L=np.array(lines).T;levels=optimize.nnls(L,profile*mask,maxiter=2000)[0];template=L@levels
  if np.linalg.norm(template)<1e-10:template=L@np.array([1/(h+1) for h in range(len(lines))])
  D.append(template*weight);anchors[p]=dict(events=chosen,fallback_includes_harmonic_collisions=fallback)
 D=np.array(D).T;D/=np.maximum(np.linalg.norm(D,axis=0),1e-12)
 outputs=np.load(MODELS/'bytedance/model_outputs.npz');onsets=outputs['reg_onset_output'];frames=outputs['frame_output'];audit=[]
 for i,n in enumerate(candidates):
  at=n['start'];p=n['pitch'];ps=sorted({m['pitch'] for m in candidates if abs(m['start']-at)<.055});ix=[pitches.index(p) for p in ps];A=D[:,ix];target=Y[i]*weight
  coef=optimize.nnls(A,target,maxiter=2000)[0];own=ps.index(p);full=np.linalg.norm(target-A@coef)**2;red=np.delete(A,own,axis=1);less=np.linalg.norm(target-red@optimize.nnls(red,target,maxiter=2000)[0])**2 if red.shape[1] else np.dot(target,target)
  lo=max(0,round((at-.035)*100));hi=min(len(onsets),round((at+.04)*100)+1);peak=float(onsets[lo:hi,p-21].max());frame=float(frames[lo:min(len(frames),hi+12),p-21].max())
  carriers=[q for q in ps if p-q in [12,19,24,28,31,36]]
  prev=[m for m in candidates if m['pitch']==p and .045<at-m['start']<=.22]
  r=dict(index=i,pitch=p,source_time=at,agreement=n['model_agreement'],source_gain=float(max(0,(less-full)/max(np.dot(target,target),1e-12))),relative_coefficient=float(coef[own]/max(coef.max(),1e-12)),byte_onset=peak,byte_frame=frame,lower_carriers=carriers,rapid_repeat_candidate=bool(prev),template_fallback=anchors[p]['fallback_includes_harmonic_collisions'])
  keep=n['model_agreement'];reason='both independent piano transcriptions agree'
  if not keep:
   if carriers:
    keep=r['source_gain']>=.035 and r['relative_coefficient']>=.25 and peak>=.20
    reason='single-model harmonic candidate requires independent attack and onset support'
   else:
    keep=(r['source_gain']>=.018 and r['relative_coefficient']>=.18 and (peak>=.16 or n['transkun_id'] is not None)) or (peak>=.70 and frame>=.70)
    reason='single-model attack supported by local source fit or strong note posterior'
   # Very fast real trills must not disappear just because a frame decoder
   # merges them; require fresh source evidence at each distinct strike.
   if prev and n['transkun_id'] is not None and r['source_gain']>=.065 and r['relative_coefficient']>=.35:
    keep=True;reason='rapid TransKun reattack with independent source spectral increment'
   if n['end']-n['start']<.020:keep=False;reason='unconfirmed sub-20ms single-model event'
  r.update(kept=bool(keep),reason=reason);audit.append(r)
 chosen=[]
 for n,r in zip(candidates,audit):
  if r['kept']:chosen.append(copy.deepcopy(n))
 chosen.sort(key=lambda n:(n['start'],n['pitch']));prev={}
 for i,n in enumerate(chosen):
  n['id']=i;n['hand']='right' if n['pitch']>=60 else 'left';n['original_velocity']=n['velocity']
  if n['pitch'] in prev:prev[n['pitch']]['end']=min(prev[n['pitch']]['end'],n['start']-.005)
  prev[n['pitch']]=n
  assert n['end']>n['start']
 old=load(OLD_PERFORMANCE)
 old_origin=ORIGIN if OLD_ORIGIN is None else OLD_ORIGIN
 for n in old:n['start']+=old_origin;n['end']+=old_origin
 mo=match(chosen,old,.10)
 report=dict(raw_transkun=len(a),raw_bytedance=len(b),shared_attacks=len(ab),single_model_candidates=len(candidates)-len(ab),kept_single_model=sum(not n['model_agreement'] for n in chosen),withheld=sum(not r['kept'] for r in audit),final_notes=len(chosen),v3_notes=len(old),matched_v3=len(mo),v3_unmatched=len(old)-len(mo),new_vs_v3=len(chosen)-len(mo),source_origin=ORIGIN,accuracy_measured=False)
 dump(W/'candidate_audit.json',audit);dump(W/'candidates.json',candidates);dump(W/'source_template_anchors.json',anchors);dump(W/'summary.json',report)
 dump(W/'v3_unmatched.json',[n for i,n in enumerate(old) if i not in set(mo.values())]);dump(W/'new_vs_v3.json',[n for i,n in enumerate(chosen) if i not in mo])
 for n in chosen:n['start']-=ORIGIN;n['end']-=ORIGIN
 pedal=[[max(0,x-ORIGIN),max(0,z-ORIGIN)] for x,z in load(MODELS/'transkun/model_pedal.json') if z>ORIGIN]
 dump(W/'performance_uncalibrated.json',chosen);dump(W/'pedal.json',pedal)
 np.savez_compressed(W/'source_attack_templates.npz',D=D,pitches=pitches,weight=weight)
 print(json.dumps(report,ensure_ascii=False),flush=True)
 print('KEPT SINGLE',[(r['pitch'],round(r['source_time'],3),round(r['source_gain'],3),round(r['byte_onset'],3),r['rapid_repeat_candidate']) for r in audit if r['kept'] and not r['agreement']])
if __name__=='__main__':main()
