"""Use the deployment's interpreter without relying on shell activation."""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / '.deployment.json').read_text())
env = os.environ.copy()
env.update(PIANO_FFMPEG=cfg['ffmpeg'], PIANO_FFPROBE=cfg['ffprobe'],
           PIANO_NODE=cfg['node'], PIANO_RSVG=cfg['rsvg'], NODE_PATH=cfg['node_modules'])
env['PYTHONPATH'] = ''
env['PIANO_EXTRA_PYTHONPATH'] = os.pathsep.join(cfg.get('extra_pythonpaths', []))
os.execve(cfg['python'], [cfg['python'], str(ROOT / 'scripts/pipeline.py'), *sys.argv[1:]], env)
