"""Run the two priority piano models once dependencies/checkpoints are available.
The checkpoint is explicit: no automatic fallback model or hidden download.
"""
import argparse,json,time,hashlib,sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('backend',choices=['transkun','bytedance']);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--device',default='cpu',choices=['cpu','cuda']);p.add_argument('--audio',type=Path,default=ROOT/'work/audio/烟花易冷.wav');p.add_argument('--output',type=Path);a=p.parse_args()
 if not a.checkpoint.is_file() or a.checkpoint.stat().st_size<1000000:p.error('A complete local official checkpoint is required; no automatic download is attempted.')
 if a.backend=='bytedance' and a.checkpoint.stat().st_size<160000000:p.error('The official ByteDance loader would download a replacement for files smaller than 160 MB; supply the complete checkpoint.')
 package='transkun' if a.backend=='transkun' else 'piano_transcription_inference'
 if importlib.util.find_spec(package) is None:p.error(f'{package} is not installed in this Python environment.')
 import torch,numpy as np
 from scipy.io import wavfile
 from scipy.signal import resample_poly
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA was requested but is not available.')
 out=a.output or ROOT/'result/烟花易冷/开源模型测试_20261006'/a.backend;out.mkdir(parents=True,exist_ok=True);mid=out/'model_raw.mid'
 if mid.exists():p.error(f'{mid} already exists; preserve it before running a new experiment.')
 sr,y=wavfile.read(a.audio);y=y.astype(np.float32)/32768 if y.dtype==np.int16 else y.astype(np.float32);y=y.mean(1) if y.ndim==2 else y;start=time.monotonic()
 if a.backend=='bytedance':
  from piano_transcription_inference import PianoTranscription
  from math import gcd
  g=gcd(sr,16000);audio=resample_poly(y,16000//g,sr//g).astype(np.float32);model=PianoTranscription(checkpoint_path=str(a.checkpoint),device=a.device);events=model.transcribe(audio,str(mid))
  np.savez_compressed(out/'model_outputs.npz',**events['output_dict'])
 else:
  import transkun,moduleconf
  config=a.config or Path(transkun.__file__).parent/'pretrained/2.0.conf';manager=moduleconf.parseFromFile(str(config));model=manager['Model'].module.TransKun(conf=manager['Model'].config).to(a.device);checkpoint=torch.load(a.checkpoint,map_location=a.device);state=checkpoint.get('best_state_dict',checkpoint.get('state_dict'))
  if state is None:raise ValueError('No recognized TransKun state dictionary in checkpoint')
  model.load_state_dict(state,strict=True);model.eval()
  from math import gcd
  g=gcd(sr,model.fs);audio=resample_poly(y,model.fs//g,sr//g).astype(np.float32);events=model.transcribe(torch.from_numpy(audio[:,None]).to(a.device),discardSecondHalf=False)
  from transkun.Data import writeMidi
  writeMidi(events).write(str(mid))
 import pretty_midi
 decoded=pretty_midi.PrettyMIDI(str(mid));notes=sorted([dict(start=n.start,end=n.end,pitch=n.pitch,velocity=n.velocity) for inst in decoded.instruments for n in inst.notes],key=lambda n:(n['start'],n['pitch']))
 for i,n in enumerate(notes):n['id']=i
 pedals=[]
 for inst in decoded.instruments:
  down=None
  for c in sorted(inst.control_changes,key=lambda x:x.time):
   if c.number!=64:continue
   if c.value>=64 and down is None:down=c.time
   elif c.value<64 and down is not None:pedals.append([down,c.time]);down=None
  if down is not None:pedals.append([down,decoded.get_end_time()])
 (out/'model_notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2));(out/'model_pedal.json').write_text(json.dumps(pedals,indent=2));report=dict(model=a.backend,device=a.device,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,notes=len(notes),pedal_intervals=len(pedals),checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),runtime_seconds=time.monotonic()-start,raw_model_output=True,manual_pitch_correction=False,accuracy_score=None)
 (out/'inference_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
