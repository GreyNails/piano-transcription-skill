"""Readable rhythm and pedal-supported notation, without changing performance."""
import json,copy,sys,subprocess,hashlib
import numpy as np
from pathlib import Path
import transcribe as t
import score_optimize as s
import deliver_local_three_v3 as d
from deliver_focused_repairs import xml_check
W=t.WORK/'ophelia_v6';OUT=t.RESULT/'TheFateofOphelia/原音对照版_v6';TITLE='TheFateofOphelia_原音对照v6'
def load(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 data=load(OUT/'score.json');notes=load(W/'delivered_performance.json');beats=np.array(load(W/'delivery_beats.json'));pd=load(W/'pedal.json')
 performance_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/(TITLE+'.mid'),OUT/(TITLE+'.mp3')]}
 if not (W/'raw_quantized_score.json').exists():dump(W/'raw_quantized_score.json',data)
 groups=[]
 for n in notes:
  if groups and n['start']-groups[-1][0]['start']<=.030:groups[-1].append(n)
  else:groups.append([n])
 qn=[];last={};shift=[]
 for g in groups:
  at=float(np.median([n['start'] for n in g]));raw=float(np.interp(at,beats,np.arange(len(beats)))*8)
  q=round(raw)
  for step,tol in [(4,.040),(2,.028)]:
   v=round(raw/step)*step;sec=float(np.interp(v/8,np.arange(len(beats)),beats))
   if abs(sec-at)<=tol:q=v;break
  # Preserve a genuinely separate same-key reattack even in a grace figure.
  q=max(q,max((last.get(n['pitch'],-1)+1 for n in g),default=0))
  for n in g:
   end=max(q+1,round(float(np.interp(n['end'],beats,np.arange(len(beats)))*8)))
   nn=dict(n,q=int(q),end_q=int(end));qn.append(nn);last[n['pitch']]=q
   shift.append(abs(float(np.interp(q/8,np.arange(len(beats)),beats))-n['start']))
 qn.sort(key=lambda n:(n['q'],n['pitch']));releases=[]
 # Finger release is not the same as written rhythmic duration under pedal.
 # Extend only within measured pedal, and only to the next written voice
 # attack within one quarter note. No sustain or attack is added to audio.
 for voice in ['melody','inner','accompaniment','bass']:
  ticks=sorted({n['q'] for n in qn if n['voice']==voice});nextq=dict(zip(ticks,ticks[1:]))
  for n in qn:
   if n['voice']!=voice:continue
   q=n['q'];end=n['end_q'];nxt=nextq.get(q)
   if nxt is not None:
    end=min(end,nxt)
    at_next=float(np.interp(nxt/8,np.arange(len(beats)),beats))
    pedal_support=any(a<=n['end']<z and z>=at_next-.035 for a,z in pd)
    if 0<nxt-q<=8 and pedal_support:end=nxt
   # Prefer sixteenth/eighth duration boundaries when within one 32nd.
   rounded=int(round(end/2)*2)
   if rounded>q and (nxt is None or rounded<=nxt) and abs(rounded-end)<=1:end=rounded
   if end!=n['end_q']:releases.append(dict(id=n['id'],from_tick=n['end_q'],to_tick=end))
   n['end_q']=max(q+1,end)
 prev={}
 for n in qn:
  if n['pitch'] in prev:prev[n['pitch']]['end_q']=min(prev[n['pitch']]['end_q'],n['q'])
  prev[n['pitch']]=n
 bar_ticks=data.get('barTicks',32);beat_ticks=data.get('notationBeatTicks',8)
 bars=int(np.ceil(max(n['end_q'] for n in qn)/bar_ticks));data['voiceMeasures']={v:d.segments(qn,v,bars,bar_ticks,beat_ticks) for v in ['melody','inner','accompaniment','bass']};hands=[d.segments([dict(n,voice=n['hand']) for n in qn],hand,bars,bar_ticks,beat_ticks) for hand in ['right','left']];data['measures']=[[hands[0][i],hands[1][i]] for i in range(bars)]
 data.update(engravingInstruction='Reading rhythm follows phrase pulse. Pedal-supported notation; MIDI retains measured key releases.',notationDurationConvention='Readable voice durations may include measured pedal sustain; original finger gates remain in MIDI/MP3.',notationReleaseEdits=releases)
 dump(OUT/'score.json',data);dump(W/'reading_score_notes.json',qn);s.xml_score(data,OUT/(TITLE+'.musicxml'))
 subprocess.run([s.NODE,str(t.ROOT/'scripts/engrave_score_edit.cjs'),str(OUT)],check=True);subprocess.run([sys.executable,str(t.ROOT/'scripts/pdf_export.py'),str(OUT)],check=True)
 xml_check(OUT/(TITLE+'.musicxml'),qn,bar_ticks);assert len(s.decode(data))==len(notes)
 for p,h in performance_hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 report=dict(note_attacks_preserved=len(notes),performance_files_unchanged=True,reading_onset_displacement_median_ms=float(np.median(shift)*1000),reading_onset_displacement_p95_ms=float(np.percentile(shift,95)*1000),reading_onset_displacement_max_ms=float(max(shift)*1000),notation_duration_adjustments=len(releases),score_pages_expected=int(np.ceil(bars/4)),pedal_source='measured TransKun CC64')
 dump(OUT/'阅读谱整理检查.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
