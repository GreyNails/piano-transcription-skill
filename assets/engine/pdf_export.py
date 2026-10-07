import os, sys, subprocess, json
from pathlib import Path
sys.path.extend(p for p in os.environ.get('PIANO_EXTRA_PYTHONPATH','').split(os.pathsep) if p)
ROOT=Path(os.environ['PIANO_RUNTIME'])
import fitz
explicit=[Path(a) for a in sys.argv[1:] if Path(a).is_absolute()]
for folder in explicit or sorted((ROOT/'result').iterdir()):
    if not folder.is_dir() or not (folder/'score.json').exists():continue
    if not explicit and len(sys.argv)>1 and folder.name not in sys.argv[1:]:continue
    metadata=json.loads((folder/'score.json').read_text());title=metadata.get('title',folder.name)
    pages=ROOT/'work/pages'/title
    if not pages.exists():continue
    doc=fitz.open()
    for svg in sorted(pages.glob('*.svg')):
        pdf=svg.with_suffix('.pdf')
        subprocess.run([os.environ['PIANO_RSVG'],'--dpi-x','72','--dpi-y','72','-f','pdf','-o',str(pdf),str(svg)],check=True)
        with fitz.open(pdf) as part:doc.insert_pdf(part)
    if not len(doc):raise RuntimeError('Empty PDF')
    doc.set_metadata({'title':metadata.get('displayTitle',title)+(' - Expressive piano arrangement' if metadata.get('arrangement') else ' - Piano transcription draft'),'subject':'Piano grand staff','creator':'VexFlow and librsvg'})
    doc.save(folder/(title+'.pdf'),garbage=4,deflate=True)
    print('PDF',folder.name,len(doc),'pages',flush=True)
