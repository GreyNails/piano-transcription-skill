"""First source-reviewed editions for the three newly supplied videos."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import argparse,copy,csv,hashlib,json,math,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
import transcribe as t
import batch_refine as visual
import batch_deliver as b
import deliver_local_three_v3 as d
import score_optimize as s
import yanhua_v4_reconcile as reconcile
import yanhua_v4_dynamics as dynamics
import yanhua_natural as synth
import ophelia_v6_reading_score as reading
from batch_validate import read_midi
from dual_source_revision import measure,command
R=t.ROOT;W=t.WORK/'add_source_20261007'
CONFIG={
 'hesapirate1':dict(title="He's a Pirate",strike=302,keyboard=355,meter='3/4',bpm=195),
 'perfect2':dict(title='Perfect',strike=395,keyboard=447,meter='12/8',bpm=100),
 'vivalavida2':dict(title='Viva La Vida',strike=328,keyboard=380,meter='4/4',bpm=140)}
def load(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def extract_visual(name):
 w=W/name;c=CONFIG[name];video=t.WORK/'videos'/f'{name}.mp4';source=Path(CONFIG[name]['source'])
 if not video.exists():video.symlink_to(source)
 assert video.resolve()==source.resolve()
 if (w/'visual.json').exists():return
 visual.W=w;visual.stage=lambda _:w
 visual.STRIKE_ROW=c['strike'];visual.KEYBOARD_ROW=c['keyboard'];visual.BAR_ROWS=[c['strike']-230,c['strike']-180,c['strike']-130]
 sr,y=wavfile.read(w/'source.wav');assert sr==t.SR
 ns,info=visual.visual_events(name,y.astype(np.float32)/32768)
 dump(w/'visual.json',dict(notes=ns,info=info));print('VISUAL READY',name,len(ns),flush=True)

def select_notes(name):
 w=W/name;dump(w/'empty_previous.json',[])
 reconcile.NAME=name;reconcile.W=w;reconcile.MODELS=w;reconcile.ORIGIN=0;reconcile.OLD_ORIGIN=0;reconcile.OLD_PERFORMANCE=w/'empty_previous.json';reconcile.SOURCE_AUDIO=w/'source.wav';reconcile.main()
 candidates=load(w/'candidates.json');audit=load(w/'candidate_audit.json');v=load(w/'visual.json')['notes'];matched=reconcile.match(candidates,v,.065);rescued=[]
 for i,j in matched.items():
  n=v[j];a=audit[i];a['visual_match']=dict(time=n['start'],rows=n['visual_rows'],attack=n['attack'])
  if not a['kept'] and n['visual_rows']>=2 and n['attack']>=.09 and candidates[i]['end']-candidates[i]['start']>=.02:
   a.update(kept=True,reason='Independent multi-row visible note trajectory plus one piano model and source-audio attack; protect genuine octave/repeat.')
   rescued.append(dict(candidate_index=i,pitch=n['pitch'],source_time=candidates[i]['start'],visual=n))
 ns=[copy.deepcopy(n) for n,a in zip(candidates,audit) if a['kept']];ns.sort(key=lambda n:(n['start'],n['pitch']));prev={}
 for i,n in enumerate(ns):
  n.update(id=i,hand='right' if n['pitch']>=60 else 'left',original_velocity=n['velocity'])
  if n['pitch'] in prev:prev[n['pitch']]['end']=min(prev[n['pitch']]['end'],n['start']-.005)
  prev[n['pitch']]=n;assert n['end']>n['start']
 origin=max(0,min(n['start'] for n in ns)-.08)
 source_ns=copy.deepcopy(ns)
 for n in ns:n['start']-=origin;n['end']-=origin
 pedal=[[max(0,a-origin),max(0,z-origin)] for a,z in load(w/'transkun/model_pedal.json') if z>origin]
 dump(w/'performance_uncalibrated.json',ns);dump(w/'pedal.json',pedal);dump(w/'source_performance.json',source_ns)
 summary=load(w/'summary.json')
 for key in ['v3_notes','matched_v3','v3_unmatched','new_vs_v3']:summary.pop(key,None)
 summary.update(final_notes=len(ns),kept_single_model=sum(not n['model_agreement'] for n in ns),withheld=sum(not a['kept'] for a in audit),
  visual_supported_restorations=len(rescued),source_origin=origin,first_edition=True,previous_version=None)
 dump(w/'summary.json',summary);dump(w/'candidate_audit.json',audit);dump(w/'visual_restored.json',rescued)
 print('SELECTED',name,json.dumps(summary,ensure_ascii=False),flush=True)

def pulse(name,ns):
 w=W/name;c=CONFIG[name];at=np.array([n['start'] for n in ns]);best=None
 for bpm in np.arange(c['bpm']*.86,c['bpm']*1.14,.01):
  step=15/bpm;z=np.mean(np.exp(2j*np.pi*at/step))
  if best is None or abs(z)>best[0]:best=(abs(z),np.angle(z)*step/(2*np.pi),step)
 strength,phase,step=best
 for _ in range(4):
  ticks=np.round((at-phase)/step);phase,step=least_squares(lambda x:at-x[0]-ticks*x[1],[phase,step],loss='soft_l1',f_scale=.015).x
 residual=abs(at-phase-np.round((at-phase)/step)*step);end=max(n['end'] for n in ns)+8
 if np.median(residual)<.018 and np.percentile(residual,95)<step*.35:
  beats=np.r_[0,phase+np.arange(1,math.ceil(end/(4*step))+2)*4*step];method='source-fitted steady quarter pulse'
 else:
  # Follow the piano's expressive tempo with a note-onset beat tracker.
  # For compound meter, track dotted quarters then subdivide to quarter units.
  import librosa
  import scipy.signal
  if not hasattr(scipy.signal,'hann'):scipy.signal.hann=scipy.signal.windows.hann
  env=np.zeros(math.ceil(end*100))
  for n in ns:env[round(n['start']*100)]+=np.sqrt(n['velocity'])
  env=gaussian_filter1d(env,1.3);compound=c['meter'].endswith('/8');factor=1.5 if compound else 1
  tempo,bf=librosa.beat.beat_track(onset_envelope=env,sr=100,hop_length=1,bpm=15/step/factor,tightness=100,trim=True)
  bb=bf/100
  if len(bb)<4:bb=np.arange(0,end,4*step*factor)
  while bb[0]>.15:bb=np.r_[bb[0]-np.median(np.diff(bb[:12])),bb]
  while bb[-1]<end:bb=np.r_[bb,bb[-1]+np.median(np.diff(bb[-12:]))]
  bb=bb-bb[0] if bb[0]<-.2 else bb
  quarters=np.arange(0,(len(bb)-1)*factor,1.0);beats=np.interp(quarters,np.arange(len(bb))*factor,bb);beats[0]=0
  method='source-adaptive dotted-quarter pulse' if compound else 'source-adaptive quarter pulse'
 assert np.all(np.diff(beats)>0)
 report=dict(method=method,bpm_quarter=15/step,phase_seconds=phase,fixed_fit_strength=strength,
  fixed_fit_median_ms=float(np.median(residual)*1000),fixed_fit_p95_ms=float(np.percentile(residual,95)*1000),meter=c['meter'],meter_inferred=not c.get('meter_explicit',False),performance_quantized=False)
 dump(w/'pulse_fit.json',report);dump(w/'delivery_beats.json',beats.tolist());return beats,report

def deliver(name):
 w=W/name;c=CONFIG[name];out=t.RESULT/name;title=name;out.mkdir(exist_ok=True)
 if (out/'最终检查.json').exists():raise RuntimeError('Complete result exists; preserve it')
 ns=load(w/'performance.json');original=copy.deepcopy(ns);pedal=load(w/'pedal.json');summary=load(w/'summary.json');origin=summary['source_origin']
 beats,pulse_info=pulse(name,ns);key,fifths,coverage=b.key_signature(ns);info=dict(key=key,fifths=fifths,bpm=round(pulse_info['bpm_quarter'],2),meter=c['meter'],diatonic_coverage=coverage)
 while beats[-1]<max([z for a,z in pedal]+[0])+4:beats=np.r_[beats,beats[-1]+np.median(np.diff(beats[-10:]))]
 data,qn,collisions=d.score(name,ns,beats,info);num,den=map(int,c['meter'].split('/'));bar_ticks=int(num*4/den*8)
 data.update(title=title,displayTitle=c['title'],bpm=info['bpm'],meter=c['meter'],barTicks=bar_ticks,notationBeatTicks=12 if den==8 else 8,
  systemsPerPage=3,systemTop=230,systemSpacing=385,rehearsalMarks={},
  performanceText='Source-timed piano performance; measured pedal and phrase dynamics.',
  footerText='TransKun / ByteDance and source-video attack review. Inferred meter; review required.',
  engravingInstruction='Reading score; audio and MIDI retain measured performance timing.')
 dump(out/'score.json',data);dump(w/'delivery_beats.json',beats.tolist());dump(w/'delivered_performance.json',ns)
 b.midi(ns,beats,pedal,out/(title+'.mid'),info)
 synth.W=w;synth.OUT=out;synth.TITLE=title;_,loudness=synth.render(ns,pedal)
 reading.W=w;reading.OUT=out;reading.TITLE=title;reading.main()
 actual=sorted(read_midi(out/(title+'.mid')),key=lambda n:(n['pitch'],n['start']));expected=sorted(ns,key=lambda n:(n['pitch'],n['start']));assert len(actual)==len(expected);errors=[]
 for a,n in zip(actual,expected):
  assert (a['pitch'],a['velocity'],a['hand'])==(n['pitch'],n['velocity'],n['hand']);errors.extend([abs(a['start']-n['start']),abs(a['end']-n['end'])])
 assert max(errors)<.002
 assert all(tuple(a[k] for k in ['pitch','start','end','velocity'])==tuple(n[k] for k in ['pitch','start','end','velocity']) for a,n in zip(ns,original))
 raw=measure(w/'natural_render.wav');gain=-18-float(raw['input_i']);filters=f'aresample=88200,volume={gain:.6f}dB,alimiter=limit=0.794328:attack=5:release=80:level=false:latency=true,aresample=44100'
 command([t.FF,'-v','error','-y','-i',str(w/'natural_render.wav'),'-af',filters,'-codec:a','libmp3lame','-b:a','256k',str(out/(title+'.mp3'))])
 master=measure(out/(title+'.mp3'));assert float(master['input_tp'])<=-1.2
 command([t.FF,'-v','error','-y','-i',str(out/(title+'.mp3')),'-ar',str(t.SR),'-c:a','pcm_f32le',str(w/'master_decoded.wav')])
 command([t.FF,'-v','error','-y','-i',str(w/'source.wav'),'-codec:a','libmp3lame','-b:a','192k',str(out/(name+'_原音频.mp3'))])
 sr,source=wavfile.read(w/'source.wav');duration=len(source)/sr;clip_length=min(16,max(.5,(duration-origin)/3));starts=np.linspace(origin,max(origin,duration-clip_length),6);windows=[];comp=out/'原音_钢琴重奏试听对比';comp.mkdir(exist_ok=True)
 for clip_index,a in enumerate(starts):
  a=float(a);z=min(a+clip_length,duration);windows.append([a,z])
  for label,p,offset in [('原音',w/'source.wav',0),('钢琴重奏',out/(title+'.mp3'),origin)]:
   command([t.FF,'-v','error','-y','-ss',str(a-offset),'-i',str(p),'-t',str(z-a),'-af','loudnorm=I=-18:TP=-1.5:LRA=20','-ar','44100','-codec:a','libmp3lame','-b:a','192k',str(comp/f'{clip_index+1:02}_原视频{a:07.3f}-{z:07.3f}秒_{label}.mp3')])
 dump(out/'transcription.json',dict(info,source=load(w/'source_info.json'),source_origin_video_seconds=origin,note_count=len(ns),method='Dual piano models, source-specific spectral attack audit; optional calibrated video support; phrase dynamics and measured pedal',first_edition=True,review_required=True))
 for src,dst in [('candidate_audit','逐音筛选依据'),('visual_restored','视频支持恢复音'),('dynamics_report','力度处理说明'),('dynamics_changes','力度调整'),('pulse_fit','节拍分析')]:dump(out/(dst+'.json'),load(w/(src+'.json')))
 rows=load(w/'candidate_audit.json')
 with (out/'逐音筛选依据.csv').open('w',encoding='utf-8-sig',newline='') as f:
  cw=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));cw.writeheader();cw.writerows(rows)
 dump(out/'模型来源.json',{model:load(w/model/'inference_report.json') for model in ['transkun','bytedance']})
 report=dict(summary,midi_max_error_ms=max(errors)*1000,pedal_intervals=len(pedal),loudness=master,comparison_video_windows=windows,performance_timing_preserved=True,subjective_listening_verified=False)
 dump(out/'修订与检查.json',report);print('DELIVERED',name,len(ns),flush=True)

def check(name):
 w=W/name;out=t.RESULT/name;c=CONFIG[name];r=load(out/'修订与检查.json');ns=load(w/'performance.json');before=load(w/'performance_uncalibrated.json');origin=r['source_origin']
 assert len(ns)==len(before)
 assert all(tuple(a[k] for k in ['pitch','start','end'])==tuple(n[k] for k in ['pitch','start','end']) and abs(a['velocity']-n['velocity'])<=12 for a,n in zip(ns,before))
 sr,source=wavfile.read(w/'source.wav');source=source.astype(float)/32768;hop=round(sr*.01)
 def env(x):
  if x.ndim==2:x=x.mean(1)
  return np.sqrt(np.mean(np.pad(x,(0,(-len(x))%hop)).reshape(-1,hop)**2,axis=1)+1e-12)
 se=env(source);continuity={}
 for tag,path in [('uncalibrated',w/'uncalibrated/natural_render.wav'),('final_mp3',w/'master_decoded.wav')]:
  rr,x=wavfile.read(path);assert rr==sr;e=env(x);start=round(origin*sr/hop);n=min(len(e),len(se)-start);a=gaussian_filter1d(se[start:start+n],6);z=gaussian_filter1d(e[:n],6)
  bad=(a>np.percentile(a,90)*.05)&(z<np.percentile(z,90)*.005);edges=np.diff(np.r_[False,bad,False].astype(int));aa=np.flatnonzero(edges==1);zz=np.flatnonzero(edges==-1)
  continuity[tag]=dict(rms_envelope_correlation=float(np.corrcoef(a,z)[0,1]),unintended_silence_over_300ms=[[float(i*hop/sr+origin),float(j*hop/sr+origin)] for i,j in zip(aa,zz) if (j-i)*hop/sr>=.3])
 prev={};fast=[]
 for n in ns:
  if n['pitch'] in prev:
   o=prev[n['pitch']];assert o['end']<=n['start']+.001
   if .04<n['start']-o['start']<=.2:fast.append(dict(pitch=n['pitch'],source_time=n['start']+origin,interval=n['start']-o['start']))
  prev[n['pitch']]=n
 import fitz
 with fitz.open(out/(name+'.pdf')) as pdf:
  pages=len(pdf)
  for page in [0,pages//2]:pdf[page].get_pixmap(matrix=fitz.Matrix(1,1)).save(w/f'score_preview_{page+1}.png')
 assert len(list((out/'原音_钢琴重奏试听对比').glob('*.mp3')))==12
 for p,h in load(W/'previous_result_hashes.json').items():assert sha(Path(p))==h,p
 assert sha(Path(c['source']))==load(w/'source_info.json')['sha256']
 reading_report=load(out/'阅读谱整理检查.json');reading_report.update(score_pages_expected=pages,systems_per_page=3,meter=c['meter']);dump(out/'阅读谱整理检查.json',reading_report)
 r.update(pdf_pages=pages,continuity=continuity,same_key_overlaps=0,rapid_attacks_40_to_200ms=len(fast),previous_files_preserved=len(load(W/'previous_result_hashes.json')),complete=True,accuracy_measured=False)
 dump(out/'最终检查.json',r);dump(out/'快速重复音.json',fast)
 (out/'版本说明.md').write_text(f'''# {c['title']} — 新视频原音对照提取

来源：`{c['source']}`。独立保存至本目录，没有替换此前相近文件名的版本。

采用 TransKun / ByteDance 本地 GPU 双模型，同音高 65 ms 内一对一匹配；分歧音根据原音频谱增量、谐波模板与起音支持筛选，结合新提取的多行音条证据保护真实八度和连弹。保留 {len(ns)} 个击打、{r['pedal_intervals']} 段模型踏板；{r['visual_supported_restorations']} 个候选由独立视频证据支持恢复。

乐句力度以原音 RMS 轮廓校准（±4 dB，力度变化最多 12）。MIDI/MP3 保留起音与松键时间；阅读谱单独整理声部和时值。谱面暂按 **{c['meter']}**，拍号、调号和小节起点为自动推定，需要演奏者复核。PDF {pages} 页，另附可编辑 MusicXML。

原视频时间 = 新版播放时间 + {origin:.6f} 秒。原音 MP3 保留原视频时间轴，试听目录提供 6 组等响度原音／钢琴重奏片段。

MIDI 最大导出时间误差 {r['midi_max_error_ms']:.3f} ms；同键重叠检查通过。力度轮廓相关与异常空断检查见「最终检查.json」，不等于音符准确率或人工听感验收。双模型和视频检测仍可能共同误识别。

复现：`scripts/add_video_transcription.py`；原始模型 MIDI、参数、源音频和检测缓存位于 `{w}`。
''')
 print('CHECKED',name,json.dumps(r,ensure_ascii=False),flush=True)

def full(name):
 w=W/name;out=t.RESULT/name
 if (out/'最终检查.json').exists():raise RuntimeError('Already complete; will not overwrite')
 extract_visual(name)
 for stage in ['select','dynamics','deliver','check']:
  marker=w/(stage+'.complete')
  if marker.exists():continue
  print('STAGE',name,stage,flush=True)
  if stage=='select':select_notes(name)
  elif stage=='dynamics':
   dynamics.W=w;dynamics.NAME=name;dynamics.SOURCE_AUDIO=w/'source.wav';dynamics.ORIGIN=load(w/'summary.json')['source_origin'];dynamics.main()
  elif stage=='deliver':deliver(name)
  else:check(name)
  marker.write_text('completed\n')
