"""Versioned, resumable video/audio -> piano MIDI, score and sampled performance."""
import argparse, contextlib, fcntl, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
sys.path.extend(p for p in os.environ.get('PIANO_EXTRA_PYTHONPATH','').split(os.pathsep) if p)

SKILL = Path(__file__).resolve().parents[1]
ENGINE = SKILL / 'assets/engine'
EXTENSIONS = {'.mp4','.mov','.mkv','.webm','.avi','.m4v','.wav','.mp3','.flac','.m4a','.aac','.ogg'}

def dump(p, data):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(p)

def load(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def run(args):
    subprocess.run(list(map(str,args)),check=True)

def probe(source):
    return json.loads(subprocess.check_output([os.environ['PIANO_FFPROBE'],'-v','error',
        '-show_format','-show_streams','-of','json',str(source)],text=True))

def packet_check(source, media_start):
    data=json.loads(subprocess.check_output([os.environ['PIANO_FFPROBE'],'-v','error','-select_streams','a:0',
        '-show_packets','-show_entries','packet=pts_time,duration_time','-of','json',str(source)],text=True))
    packets=[p for p in data['packets'] if 'pts_time' in p];gaps=[]
    for a,b in zip(packets,packets[1:]):
        end=float(a['pts_time'])+float(a.get('duration_time',0));start=float(b['pts_time'])
        if start-end>.035:gaps.append([max(0,end-media_start),max(0,start-media_start)])
    return dict(packets=len(packets),gaps_over_35ms=gaps,timeline='relative to media start; gaps preserved')

def worker(config_path):
    c=load(config_path);job=config_path.parent;name=c['name'];source=Path(c['source'])
    os.environ.update(PIANO_RUNTIME=str(job),PIANO_RESULT=str(job/'staging'))
    for path in [job/'work',job/'staging',job/'tools',job/'work/videos']:path.mkdir(parents=True,exist_ok=True)
    for target,path in [(ENGINE,job/'scripts'),(SKILL/'assets/samples',job/'tools/samples')]:
        if not path.exists():path.symlink_to(target,target_is_directory=True)
    sys.path.insert(0,str(ENGINE))
    import numpy as np
    from scipy.io import wavfile
    import torch
    import add_video_transcription as engine
    import yanhua_v4_dynamics as dynamics
    import batch_refine as visual
    engine.W=job/'work';w=job/'work'/name;w.mkdir(exist_ok=True)
    engine.CONFIG={name:dict(c,source=str(source),title=c['title'],bpm=c['bpm'] or 120,
                            meter=c['meter'],meter_explicit=c['meter_explicit'])}
    dump(job/'work/previous_result_hashes.json',c['previous_hashes'])
    if sha(source)!=c['source_sha256']:raise RuntimeError('Source changed; create a new run')
    metadata=c['media'];start=float(metadata.get('format',{}).get('start_time',0) or 0)
    if not (w/'source.wav').exists():
        temp=w/'source.partial.wav'
        run([os.environ['PIANO_FFMPEG'],'-v','error','-y','-copyts','-start_at_zero','-i',source,'-map','0:a:0',
             '-vn','-af','aresample=22050:async=1:first_pts=0','-ac','1','-ar','22050','-c:a','pcm_s16le',temp])
        temp.replace(w/'source.wav')
    sr,audio=wavfile.read(w/'source.wav')
    if len(audio)<sr*3 or np.max(np.abs(audio.astype(float)))<10:
        raise RuntimeError('Input is silent or shorter than 3 seconds; provide a longer piano excerpt')
    dump(w/'source_info.json',dict(path=str(source),sha256=c['source_sha256'],ffprobe=metadata,media_start_seconds=start))
    gaps=packet_check(source,start);dump(w/'audio_packet_check.json',gaps)
    device=c['device']
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    for model,checkpoint in [('transkun','transkun-2.0.pt'),('bytedance','bytedance.pth')]:
        out=w/model
        if (out/'inference_report.json').exists():continue
        # An interrupted inference may have a partial MIDI; retain it before retrying.
        if out.exists() and any(out.iterdir()):out.rename(w/(model+'_interrupted_'+str(time.time_ns())))
        args=[sys.executable,ENGINE/'test_additional_amt_yanhua.py',model,'--checkpoint',SKILL/'assets/models'/checkpoint,
              '--audio',w/'source.wav','--output',out,'--device',device]
        if model=='transkun':args+=['--config',SKILL/'assets/models/transkun-2.0.conf']
        print('INFERENCE',name,model,device,flush=True);run(args)
    if not (w/'visual.json').exists():
        if c['visual']:
            v=c['visual'];engine.CONFIG[name].update(v)
            engine.extract_visual(name)
            result=load(w/'visual.json')
            if not result['notes'] or result['info']['motion_row_correlation']<.45:
                result['info']['disabled_reason']='Insufficient motion consistency; no video-based recovery applied'
                result['notes']=[];dump(w/'visual.json',result)
        else:dump(w/'visual.json',dict(notes=[],info={'enabled':False,'reason':'No calibrated falling-note geometry supplied'}))
    if not (w/'select.complete').exists():
        engine.select_notes(name);(w/'select.complete').write_text('completed\n')
    notes=load(w/'performance_uncalibrated.json')
    if not notes:raise RuntimeError('No piano notes survived; input may not be solo piano')
    if not c['bpm']:
        # Start from the onset envelope. The source-fitting stage refines this
        # prior; no inferred tempo ever changes performance onset timestamps.
        import librosa
        import scipy.signal
        if not hasattr(scipy.signal,'hann'):scipy.signal.hann=scipy.signal.windows.hann
        end=max(n['end'] for n in notes)+2;envelope=np.zeros(int(end*100)+1)
        for n in notes:envelope[round(n['start']*100)]+=np.sqrt(n['velocity'])
        from scipy.ndimage import gaussian_filter1d
        estimate,_=librosa.beat.beat_track(onset_envelope=gaussian_filter1d(envelope,1.3),sr=100,hop_length=1)
        bpm=float(np.asarray(estimate).reshape(-1)[0]);factor=1.5 if c['meter'].endswith('/8') else 1
        engine.CONFIG[name]['bpm']=max(40,min(240,bpm*factor)) if bpm>0 else 120
    if not (w/'dynamics.complete').exists():
        dynamics.W=w;dynamics.NAME=name;dynamics.SOURCE_AUDIO=w/'source.wav'
        dynamics.ORIGIN=load(w/'summary.json')['source_origin'];dynamics.MASK_INTERVALS=gaps['gaps_over_35ms']
        dynamics.main();(w/'dynamics.complete').write_text('completed\n')
    if not (w/'deliver.complete').exists():
        engine.deliver(name);(w/'deliver.complete').write_text('completed\n')
    if not (w/'check.complete').exists():
        engine.check(name);(w/'check.complete').write_text('completed\n')
    out=job/'staging'/name;dest=Path(c['destination'])
    report=load(out/'最终检查.json')
    report.update(video_evidence_used=bool(load(w/'visual.json')['notes']),audio_packet_check=gaps,
                  meter_explicit=c['meter_explicit'],source_media_start_seconds=start,
                  end_to_end_automated=True,manual_listening_verified=False)
    dump(out/'最终检查.json',report)
    dump(out/'运行参数.json',{k:v for k,v in c.items() if k not in ['previous_hashes','media']})
    dump(out/'原音频数据完整性检查.json',gaps)
    (out/'版本说明.md').write_text(f'''# {c['title']} — 钢琴原音对照提取

输入：`{source}`，SHA-256：`{c['source_sha256']}`。

本地 TransKun + ByteDance 双模型识别；65 ms 同音高匹配；单模型分歧依据原音谐波模板、频谱增量和起音概率筛选。独立视频支持：{report['video_evidence_used']}。

共 {report['final_notes']} 个音符击打，PDF {report['pdf_pages']} 页。MIDI、重奏 MP3 保留提取的实际起止时间与踏板；力度轮廓最多 ±4 dB、单音力度变化最多 12。阅读谱独立量化；拍号 {c['meter']}（{'用户指定' if c['meter_explicit'] else '默认待复核'}），不代表人工逐音校订。

`{name}.mp3` 是钢琴重奏；`{name}_原音频.mp3` 是源音轨。原媒体相对时间 = 重奏播放时间 + {report['source_origin']:.6f} 秒。试听目录提供 6 组对照片段。

MIDI 最大导出误差 {report['midi_max_error_ms']:.3f} ms。完整逐音、谱面、力度及连续性检查见同目录 JSON/CSV。数值检查不等于音符准确率、结构准确性或人工听感验收。源文件缺口见「原音频数据完整性检查.json」，不会通过移动后续音符掩盖缺口。

运行参数见「运行参数.json」。中间文件与日志：`{job}`。部署和使用说明见 skill 的 `SKILL.md` 与 `references/deployment.md`。
''')
    if dest.exists():raise RuntimeError('Output version already exists; refusing overwrite: '+str(dest))
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(out),str(dest));dump(job/'published.json',dict(destination=str(dest),complete=True))
    print('COMPLETE',dest,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',nargs='?',type=Path)
    p.add_argument('--output',type=Path,default=Path.cwd()/'result')
    p.add_argument('--work-dir',type=Path,default=SKILL/'.runs')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--bpm',type=float,help='Quarter-note BPM prior; performance is never tempo-quantized')
    p.add_argument('--meter',choices=['2/4','3/4','4/4','6/8','9/8','12/8'])
    p.add_argument('--visual-config',type=Path,help='JSON mapping input stems to full-88-key falling-note geometry')
    p.add_argument('--resume',action='store_true',help='Resume an exact source/options match, or skip it if already complete')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:return worker(a.worker)
    if not a.input:p.error('input is required')
    source=a.input.expanduser().resolve();output=a.output.expanduser().resolve();work=a.work_dir.expanduser().resolve()
    if not source.exists():p.error('input does not exist')
    files=sorted(x for x in source.iterdir() if x.suffix.lower() in EXTENSIONS and x.is_file()) if source.is_dir() else [source]
    if not files:p.error('no supported media files found')
    if a.bpm is not None and not 30<=a.bpm<=300:p.error('--bpm must be between 30 and 300')
    visual=load(a.visual_config) if a.visual_config else {}
    work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(b''.join(x.read_bytes() for x in sorted(ENGINE.glob('*')) if x.is_file())+
        Path(__file__).read_bytes()+(SKILL/'assets/checksums.json').read_bytes()+(SKILL/'requirements.lock').read_bytes()).hexdigest()
    results=[];failures=[]
    # One process owns version allocation and publication for each output root.
    with (output/'.piano-transcription.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for file in files:
            try:
                name=re.sub(r'[^\w.\-\u4e00-\u9fff]+','_',file.stem).strip('.') or 'piano'
                v=visual.get(file.stem)
                if v:
                    if set(v)-{'strike','keyboard','title'}:raise ValueError('Unknown visual geometry field')
                    if not (230<=v['strike']<=580 and v['strike']<v['keyboard']<=605):raise ValueError('Invalid visual rows at normalized 1088 x 612')
                params=dict(source=str(file),source_sha256=sha(file),name=name,title=(v or {}).get('title',file.stem),
                            device=a.device,bpm=a.bpm,meter=a.meter or '4/4',meter_explicit=bool(a.meter),visual=v,engine_sha256=fingerprint)
                signature=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
                matches=[x for x in work.glob('*/config.json') if load(x).get('signature')==signature and str(Path(load(x)['destination']).parents[1])==str(output)]
                if a.resume and matches:
                    config=max(matches,key=lambda x:x.stat().st_mtime);job=config.parent
                    if (job/'published.json').exists():
                        destination=Path(load(job/'published.json')['destination'])
                        if not (destination/'最终检查.json').exists():raise RuntimeError('Published result was moved or deleted; rerun without --resume')
                        results.append(str(destination));print('SKIP COMPLETE',destination,flush=True);continue
                else:
                    versions=[int(x.name.rsplit('_v',1)[1]) for x in (output/name).glob('原音对照版_v*') if x.name.rsplit('_v',1)[1].isdigit()]
                    # Include reserved, interrupted jobs to avoid choosing their destination.
                    for previous in work.glob('*/config.json'):
                        d=Path(load(previous)['destination'])
                        if d.parent==output/name and d.name.rsplit('_v',1)[-1].isdigit():versions.append(int(d.name.rsplit('_v',1)[-1]))
                    dest=output/name/f'原音对照版_v{max(versions,default=0)+1}'
                    job=work/(name+'_'+str(time.time_ns()));job.mkdir()
                    before={str(x):sha(x) for x in (output/name).rglob('*') if x.is_file()}
                    metadata=probe(file)
                    if not any(x['codec_type']=='audio' for x in metadata['streams']):raise RuntimeError('No audio stream')
                    if v and not any(x['codec_type']=='video' for x in metadata['streams']):raise ValueError('Video geometry supplied for audio-only input')
                    params.update(signature=signature,destination=str(dest),previous_hashes=before,media=metadata)
                    config=job/'config.json';dump(config,params)
                print('PROCESS',file,'LOG',job/'pipeline.log',flush=True)
                with (job/'pipeline.log').open('a') as log:
                    status=subprocess.run([sys.executable,__file__,'--worker',str(config)],stdout=log,stderr=subprocess.STDOUT)
                if status.returncode:
                    print((job/'pipeline.log').read_text()[-5000:],file=sys.stderr)
                    raise RuntimeError(f'Failed; logs retained at {job}; retry with --resume')
                results.append(load(job/'published.json')['destination'])
                print('SAVED',results[-1],flush=True)
            except Exception as e:
                failures.append(dict(source=str(file),error=str(e)));print('FAILED',file,e,file=sys.stderr,flush=True)
    report=work/('run_'+str(time.time_ns())+'.json');dump(report,dict(results=results,failures=failures))
    print('RUN REPORT',report,flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
