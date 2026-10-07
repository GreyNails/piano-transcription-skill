"""Restore missing release assets and verify each file before installation."""
import argparse, hashlib, json, os, shutil, subprocess, tarfile, tempfile, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def ensure_assets(root=ROOT,bundle=None,allow_download=True):
    expected=json.loads((root/'assets/checksums.json').read_text())
    missing=[rel for rel in expected if not (root/rel).is_file()]
    if not missing:return
    if not allow_download:
        raise RuntimeError('Missing bundled assets: '+', '.join(missing)+'. Run deploy.sh first.')
    with tempfile.TemporaryDirectory(prefix='piano-assets-') as tmp:
        if bundle is None:
            manifest=root/'assets/release.json'
            if not manifest.exists():raise RuntimeError('Missing model files. Download the complete release archive or run fetch_assets.py --bundle ARCHIVE.')
            config=json.loads(manifest.read_text());repo=config['repository'];tag=config['tag'];asset=config['asset']
            if '/' in asset or '..' in asset:raise RuntimeError('Invalid release asset filename')
            archive=Path(tmp)/asset
            gh=shutil.which('gh') or os.environ.get('PIANO_GH')
            print('Fetching verified assets from',repo,tag,flush=True)
            downloaded=False
            if gh:
                result=subprocess.run([gh,'release','download',tag,'--repo',repo,'--pattern',asset,'--dir',tmp])
                downloaded=result.returncode==0 and archive.exists()
            if not downloaded:
                url=f'https://github.com/{repo}/releases/download/{tag}/{asset}'
                try:
                    request=urllib.request.Request(url,headers={'User-Agent':'piano-transcription-skill'})
                    with urllib.request.urlopen(request,timeout=120) as response, archive.open('wb') as output:
                        shutil.copyfileobj(response,output,8*1024*1024)
                except Exception as error:
                    raise RuntimeError('Cannot download release assets. For a private repository run gh auth login --web, or download the full release archive in your browser and pass --bundle. '+str(error)) from error
            bundle=archive
        # Only explicitly named, checksum-pinned regular files can be restored.
        with tarfile.open(bundle,'r:gz') as tar:
            for rel in missing:
                target=root/rel
                if target.resolve().is_relative_to(root.resolve()) is False:raise RuntimeError('Unsafe asset path')
                member=tar.getmember('piano-transcription/'+rel)
                if not member.isfile():raise RuntimeError('Expected a regular release asset: '+rel)
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_name(target.name+'.partial')
                try:
                    with tar.extractfile(member) as source,temporary.open('wb') as output:shutil.copyfileobj(source,output,8*1024*1024)
                    if digest(temporary)!=expected[rel]:raise RuntimeError('Asset checksum mismatch: '+rel)
                    temporary.replace(target)
                finally:
                    if temporary.exists():temporary.unlink()
                print('Restored',rel,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path)
    args=parser.parse_args();ensure_assets(bundle=args.bundle)
