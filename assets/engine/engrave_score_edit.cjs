const fs=require('fs'),path=require('path');
const root=process.env.PIANO_RUNTIME;
const {JSDOM}=require('jsdom');
const dom=new JSDOM('<html><body></body></html>');global.window=dom.window;global.document=dom.window.document;
dom.window.HTMLCanvasElement.prototype.getContext=function(){return {font:'',measureText:t=>({width:[...t].reduce((a,c)=>a+(c.charCodeAt(0)>255?13:7),0)})}};
const V=require('vexflow');
function engrave(folder){
 const data=JSON.parse(fs.readFileSync(path.join(folder,'score.json'),'utf8'));
 const dir=path.join(root,'work/pages',data.title);fs.mkdirSync(dir,{recursive:true});
 for(const f of fs.readdirSync(dir))if(/^\d{3}\.(svg|pdf)$/.test(f))fs.unlinkSync(path.join(dir,f));
 const spell=data.pitchSpellings;
 const meter=(data.meter||'4/4').split('/').map(Number);
 const key=p=>{const [letter,alter]=spell[p%12];const pc={C:0,D:2,E:4,F:5,G:7,A:9,B:11}[letter]+alter;return `${letter.toLowerCase()}${alter===-1?'b':alter===1?'#':''}/${Math.floor((p-pc)/12)-1}`;};
 const columns=data.barsPerSystem||2,systems=data.systemsPerPage||4,barsPerPage=columns*systems;
 const pages=Math.ceil(data.measures.length/barsPerPage);
 for(let page=0;page<pages;page++){
  const div=document.createElement('div');document.body.appendChild(div);
  const renderer=new V.Renderer(div,V.Renderer.Backends.SVG);renderer.resize(1080,1527);const ctx=renderer.getContext();
  ctx.setFont('sans-serif',25,'bold');ctx.fillText(data.displayTitle,55,54);
  ctx.setFont('sans-serif',13,'normal');ctx.fillText(`${data.reconstructed?'Reconstructed piano score':'Edited piano score'}  |  quarter note = ca. ${data.bpm}  |  Cantabile, legato`,55,81);
  ctx.fillText(data.engravingInstruction||'Separate melody and bass voices; accompaniment lighter. Pedal with harmony.',55,103);
  for(let row=0;row<systems;row++){
   const first=page*barsPerPage+row*columns;if(first>=data.measures.length)break;
   const y=(data.systemTop||139)+row*(data.systemSpacing||1228/systems);const previous={};
   for(let col=0;col<columns;col++){
    const bar=first+col;if(bar>=data.measures.length)break;
    const w=974/columns,x=55+col*w;
    const staves=[new V.Stave(x,y,w),new V.Stave(x,y+(data.staffGap||140),w)];
    staves.forEach((st,h)=>{if(col===0){st.addClef(h?'bass':'treble');if(data.key&&data.key!=='C')st.addKeySignature(data.key);if(bar===0)st.addTimeSignature(data.meter||'4/4');}if(bar===data.measures.length-1)st.setEndBarType(V.Barline.type.END);st.setContext(ctx).draw();});
    if(col===0){new V.StaveConnector(...staves).setType('brace').setContext(ctx).draw();new V.StaveConnector(...staves).setType('singleLeft').setContext(ctx).draw();}
    ctx.setFont('sans-serif',11,'normal');ctx.fillText(String(bar+1),x+4,y+11);
    if(data.rehearsalMarks[String(bar)]){ctx.setFont('sans-serif',12,'bold');ctx.fillText(`[${data.rehearsalMarks[String(bar)]}]`,x+34,y+11);}
    const allVoices=[],byStaff=[[],[]],drawables=[],ties=[],beams=[];const melody=[];
    for(let h=0;h<2;h++){
     const names=data.staves[h];let active=names.filter(v=>data.voiceMeasures[v][bar].some(s=>s.pitches.length));if(!active.length)active=[names[0]];
     const poly=active.length>1;
     for(const voiceName of active){
      const primary=voiceName==='melody'||voiceName==='accompaniment';let tick=0;const notes=[];const segs=data.voiceMeasures[voiceName][bar];
      for(const seg of segs){
       const unit=data.ticksPerQuarter||4;
       const rest=!seg.pitches.length,dotted=(unit===8?[3,6,12,24,48]:[3,6,12]).includes(seg.ticks);
       const duration=String((unit*4)/(dotted?seg.ticks/1.5:seg.ticks))+(rest?'r':'');
       const restkey=h?(poly?(primary?'f/3':'f/2'):'d/3'):(poly?(primary?'d/5':'e/4'):'b/4');
       const n=new V.StaveNote({clef:h?'bass':'treble',keys:rest?[restkey]:seg.pitches.map(key),duration,dots:dotted?1:0,auto_stem:!poly,stem_direction:primary?V.Stem.UP:V.Stem.DOWN});
       if(dotted)V.Dot.buildAndAttach([n],{all:true});
       if(!rest){
        if(previous[voiceName]&&seg.tied.length){const prev=previous[voiceName],pairs=seg.tied.filter(p=>prev.pitches.includes(p));if(pairs.length)ties.push(new V.StaveTie({first_note:prev.note,last_note:n,first_indices:pairs.map(p=>prev.pitches.indexOf(p)),last_indices:pairs.map(p=>seg.pitches.indexOf(p))}));}
        else if(seg.tied.length)ties.push(new V.StaveTie({last_note:n,first_indices:seg.tied.map(p=>seg.pitches.indexOf(p)),last_indices:seg.tied.map(p=>seg.pitches.indexOf(p))}));
        previous[voiceName]={note:n,pitches:seg.pitches};
       }else previous[voiceName]=null;
       if(voiceName==='melody')melody.push({note:n,seg,q:tick});
       notes.push(n);tick+=seg.ticks;
      }
      // Complete a tie on both sides of a system/page break.
      if(col===columns-1&&bar+1<data.measures.length&&previous[voiceName]){
       const next=data.voiceMeasures[voiceName][bar+1][0],prev=previous[voiceName];
       const pairs=next.tied.filter(p=>prev.pitches.includes(p));
       if(pairs.length)ties.push(new V.StaveTie({first_note:prev.note,first_indices:pairs.map(p=>prev.pitches.indexOf(p)),last_indices:pairs.map(p=>prev.pitches.indexOf(p))}));
      }
      const voice=new V.Voice({num_beats:meter[0],beat_value:meter[1]}).addTickables(notes);allVoices.push(voice);byStaff[h].push(voice);drawables.push({voice,h});
      beams.push(...V.Beam.generateBeams(notes,{groups:[meter[1]===8?new V.Fraction(3,8):new V.Fraction(1,4)],stem_direction:poly?(primary?V.Stem.UP:V.Stem.DOWN):undefined}));
     }
    }
    byStaff.forEach(voices=>V.Accidental.applyAccidentals(voices,data.key||'C'));
    const start=Math.max(...staves.map(st=>st.getNoteStartX()));staves.forEach(st=>st.setNoteStartX(start));
    const formatter=new V.Formatter();byStaff.forEach(voices=>formatter.joinVoices(voices));formatter.format(allVoices,x+w-start-24);
    drawables.forEach(({voice,h})=>voice.draw(ctx,staves[h]));beams.forEach(beam=>beam.setContext(ctx).draw());ties.forEach(t=>t.setContext(ctx).draw());
    // Short editorial slurs clarify connected melodic gestures, without crossing rests.
    let run=[];const runs=[];
    for(const item of melody){if(!item.seg.pitches.length){if(run.length>=3)runs.push(run);run=[];}else if(!item.seg.tied.length)run.push(item);}
    if(run.length>=3)runs.push(run);
    for(const r of runs){if(data.suppressRepeatedNoteSlurs&&r.some((n,i)=>i>0&&n.seg.pitches.some(p=>r[i-1].seg.pitches.includes(p))))continue;if(r.length<=7&&r[r.length-1].q-r[0].q>=4&&Math.max(...r.flatMap(n=>n.seg.pitches))-Math.min(...r.flatMap(n=>n.seg.pitches))<=12){new V.Curve(r[0].note,r[r.length-1].note,{cps:[{x:0,y:10},{x:0,y:10}],position:V.Curve.Position.NEAR_TOP,position_end:V.Curve.Position.NEAR_TOP,y_shift:5}).setContext(ctx).draw();}}
    if(bar===0){ctx.setFont('serif',15,'italic');ctx.fillText('mp; melody cantabile',start+8,y+118);}
   }
  }
  ctx.setFont('sans-serif',12,'normal');ctx.fillText(data.footerText||(data.reconstructed?'Reconstructed from video note tracks. Previous editions retained.  Ped. sim.':'Editorial edition of the existing transcription. Original version retained.  Ped. sim.'),55,1460);ctx.fillText(`${page+1} / ${pages}`,965,1460);
  let svg=div.innerHTML.replace('<svg ','<svg xmlns="http://www.w3.org/2000/svg" ').replace('width="1080"','width="210mm"').replace('height="1527"','height="297mm"');
  fs.writeFileSync(path.join(dir,`${String(page+1).padStart(3,'0')}.svg`),svg);div.remove();
 }
 console.log('EDITED SCORE',data.displayTitle,pages);
}
for(const folder of process.argv.slice(2))engrave(folder);
