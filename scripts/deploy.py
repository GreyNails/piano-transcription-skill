"""Idempotent Linux deployment. Reuse verified local tools, or create a venv."""
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent

def run(args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str,args)), check=True, **kwargs)

def tool(name, candidates=(), minimum=None):
    values = [os.environ.get('PIANO_'+name.upper()), shutil.which(name), *candidates]
    for value in values:
        if not value or not Path(value).is_file(): continue
        if minimum:
            try:
                version = subprocess.check_output([str(value),'--version'],text=True).strip()
                if int(version.lstrip('v').split('.')[0]) < minimum: continue
            except (ValueError,subprocess.SubprocessError): continue
        return str(Path(value).resolve())
    raise RuntimeError(f'Missing {name}'+(f' >= {minimum}' if minimum else '')+
        '. Install the prerequisites in references/deployment.md, or use the supplied Dockerfile.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fresh',action='store_true',help='Create skill/.venv instead of using the existing piano environment')
    p.add_argument('--cpu',action='store_true',help='Install CPU-only PyTorch in fresh mode')
    p.add_argument('--python',help='Existing Python 3.11 executable; does not modify its packages')
    p.add_argument('--check',action='store_true',help='Validate deployment and assets without installing')
    p.add_argument('--docker',action='store_true',help='Build the self-contained Linux container, including system dependencies')
    a=p.parse_args()
    from fetch_assets import ensure_assets
    ensure_assets(allow_download=not a.check)
    for rel,expected in json.loads((ROOT/'assets/checksums.json').read_text()).items():
        actual=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
        if actual!=expected: raise RuntimeError('Asset checksum mismatch: '+rel)
    if a.docker:
        run(['docker','build','--build-arg','TORCH_VARIANT='+('cpu' if a.cpu else 'cu121'),
             '-t','piano-transcription-skill:local',ROOT])
        run(['docker','run','--rm','--entrypoint','/app/deploy.sh','piano-transcription-skill:local','--check'])
        print('CONTAINER VERIFIED. Run ./run-docker.sh INPUT OUTPUT [pipeline options]')
        return
    old=ROOT/'.deployment.json'
    if a.check or (old.exists() and not a.fresh and not a.python):
        if not old.exists(): raise RuntimeError('Run deploy.sh first')
        cfg=json.loads(old.read_text())
    else:
        base=PROJECT.parent
        ff=tool('ffmpeg',[base/'anaconda3/envs/acestep/bin/ffmpeg'])
        probe=tool('ffprobe',[base/'anaconda3/envs/acestep/bin/ffprobe'])
        node=tool('node',list((base/'.vscode-server/cli/servers').glob('*/server/node')),18)
        rsvg=tool('rsvg-convert')
        extra=[]
        python=Path(a.python).absolute() if a.python else ROOT/'.venv/bin/python'
        if not a.python and not a.fresh and not python.exists():
            candidate=PROJECT/'.venv-amt/bin/python'
            if candidate.exists():
                python=candidate
                pymupdf=base/'anaconda3/envs/minimaxH3/lib/python3.11/site-packages'
                if (pymupdf/'fitz').exists() or (pymupdf/'pymupdf').exists(): extra.append(str(pymupdf))
        # Keep the venv executable path: resolving it would lose its site-packages.
        if not python.exists():
            if sys.version_info[:2] != (3,11): raise RuntimeError('Fresh deployment requires Python 3.11')
            run([sys.executable,'-m','venv',ROOT/'.venv'])
        if (a.fresh or python==ROOT/'.venv/bin/python') and not a.python:
            run([python,'-m','pip','install','--upgrade','pip'])
            index='https://download.pytorch.org/whl/'+('cpu' if a.cpu else 'cu121')
            run([python,'-m','pip','install','torch==2.5.1','torchaudio==2.5.1','--index-url',index])
            run([python,'-m','pip','install','-r',ROOT/'requirements.txt'])
        modules=ROOT/'node_modules'
        if not (modules/'vexflow/package.json').exists() and (ROOT/'assets/node_modules.tar.gz').exists():
            with tarfile.open(ROOT/'assets/node_modules.tar.gz') as archive:
                for member in archive.getmembers():
                    if member.name.startswith('/') or '..' in Path(member.name).parts or not (member.isdir() or member.isfile()):
                        raise RuntimeError('Unsafe bundled Node dependency archive member')
                archive.extractall(ROOT)
        if not (modules/'vexflow/package.json').exists():
            candidate=base/'vis_piano/node_modules'
            if not a.fresh and (candidate/'jsdom/package.json').exists(): modules=candidate
            else:
                npm=shutil.which('npm')
                if not npm: raise RuntimeError('npm is required for fresh deployment')
                env=os.environ.copy();env['PATH']=str(Path(node).parent)+os.pathsep+env['PATH']
                # Debian's npm keeps dependencies outside its own directory.
                # A separately installed Node does not inherit Debian's patched search path.
                env['NODE_PATH']=os.pathsep.join([str(x) for x in [Path('/usr/share/nodejs'),Path('/usr/lib/nodejs')] if x.exists()]+[env.get('NODE_PATH','')])
                run([node,npm,'ci' if (ROOT/'package-lock.json').exists() else 'install','--no-audit','--no-fund'],cwd=ROOT,env=env)
        cfg=dict(python=str(python),ffmpeg=ff,ffprobe=probe,node=node,rsvg=rsvg,
                 node_modules=str(modules),extra_pythonpaths=extra,mode='fresh' if a.fresh else 'local')
    env=os.environ.copy();env['PYTHONPATH']='';env['NODE_PATH']=cfg['node_modules']
    check="import sys;sys.path.extend("+repr(cfg['extra_pythonpaths'])+");import torch,torchaudio,transkun,piano_transcription_inference,librosa,numpy,scipy,fitz,moduleconf; print('Python dependencies OK; CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    run([cfg['python'],'-c',check],env=env)
    run([cfg['node'],'-e',"require('jsdom');require('vexflow');console.log('Engraving dependencies OK')"],env=env)
    for key in ['ffmpeg','ffprobe']:
        run([cfg[key],'-version'],stdout=subprocess.DEVNULL)
    run([cfg['rsvg'],'--version'],stdout=subprocess.DEVNULL)
    if not a.check:
        temporary=ROOT/'.deployment.json.tmp';temporary.write_text(json.dumps(cfg,indent=2));temporary.replace(old)
    print('DEPLOYMENT VERIFIED:',ROOT,flush=True)
    print('Run: ./run.sh INPUT --output OUTPUT_DIRECTORY')

if __name__=='__main__':
    try: main()
    except (RuntimeError,subprocess.CalledProcessError) as e:
        print('DEPLOYMENT FAILED:',e,file=sys.stderr);sys.exit(1)
