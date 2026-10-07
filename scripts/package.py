"""Create a relocatable skill archive, including weights but no machine configuration."""
import hashlib,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
destination=ROOT.parent/'piano-transcription-skill.tar.gz'
excluded={'.git','.asset-cache','.venv','.runs','node_modules','validation','__pycache__'}
with tarfile.open(destination,'w:gz',compresslevel=3) as archive:
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if any(p in excluded for p in rel.parts) or path.name.startswith('.deployment.json') or path.suffix=='.pyc':continue
        if path.is_file():archive.add(path,arcname=str(Path('piano-transcription')/rel),recursive=False)
digest=hashlib.sha256(destination.read_bytes()).hexdigest()
destination.with_suffix(destination.suffix+'.sha256').write_text(digest+'  '+destination.name+'\n')
print(destination,digest)
