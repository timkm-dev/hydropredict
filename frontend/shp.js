import { parseTelemetry } from './data.js';
import { scoreEvents } from './inference.js';

const $ = id => document.getElementById(id);
let records = [], run = null, model = null, events = [], selected = null, page = 0, busy = false;
const date = t => new Date(t).toISOString().replace('T', ' ').slice(0, 16);
const number = n => Number(n).toLocaleString(undefined, {maximumFractionDigits: 4});
function status(message, error = false) { $('message').textContent = message; $('message').classList.toggle('error', error); }
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {cache:'no-store'} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if (response.status === 401) { location.assign('/auth.html'); throw new Error('Please sign in.'); }
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed. Please try again.');
  return result;
}
function lock(value) {
  busy = value;
  for (const id of ['file','history','save-review']) $(id).disabled = value;
  $('assess').disabled = value || !records.length || !model;
  $('assess').textContent = value ? 'Assessing…' : 'Run assessment';
}
function latestReview(event) { return run.reviews.filter(r => r.episode === event.time).at(-1); }
function filteredRows(rows) {
  if ($('period').value === 'all' || !records.length) return rows;
  const from = Date.parse(records.at(-1).t) - 86400000;
  return rows.filter(r => Date.parse(r.t) >= from);
}
function chart(rows, key, threshold) {
  if (!rows.length) return '<p>No assessed readings in this period. Try a longer period or a continuous recording.</p>';
  const width=960, height=210, left=62, right=20, top=18, bottom=35;
  const times=rows.map(r=>Date.parse(r.t)), values=rows.map(r=>r[key]);
  let low=Infinity, high=-Infinity;
  for (const value of values) { low=Math.min(low,value); high=Math.max(high,value); }
  if (threshold !== undefined) { low=Math.min(low,threshold); high=Math.max(high,threshold); }
  const pad=(high-low)*.12 || 1; low-=pad; high+=pad;
  const x=t=>left+(t-times[0])/(times.at(-1)-times[0] || 1)*(width-left-right);
  const y=v=>top+(high-v)/(high-low)*(height-top-bottom);
  let path='';
  for(let i=0;i<rows.length;i++) {
    const gap=i && (times[i]-times[i-1]>450000 || times[i]-times[i-1]<150000);
    path+=`${i===0 || gap?'M':'L'}${x(times[i]).toFixed(2)},${y(values[i]).toFixed(2)} `;
  }
  const dots=threshold === undefined ? '' : rows.filter(r=>r.is_alert).map(r=>`<circle cx="${x(Date.parse(r.t)).toFixed(2)}" cy="${y(r[key]).toFixed(2)}" r="2.6" fill="#b57924"/>`).join('');
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${key === 'anomaly_score'?'Anomaly score':'Measurement'} over time; gaps break the line"><line x1="${left}" x2="${width-right}" y1="${height-bottom}" y2="${height-bottom}" stroke="#dbe3dc"/><text x="4" y="25" font-size="11" fill="#63746a">${number(high-pad)}</text><text x="4" y="${height-bottom}" font-size="11" fill="#63746a">${number(low+pad)}</text>${threshold===undefined?'':`<line x1="${left}" x2="${width-right}" y1="${y(threshold)}" y2="${y(threshold)}" stroke="#ae7529" stroke-dasharray="6 5"/>`}<path d="${path}" fill="none" stroke="#28624f" stroke-width="1.5"/>${dots}<text x="${left}" y="202" font-size="11" fill="#63746a">${date(rows[0].t)}</text><text x="${width-right}" y="202" text-anchor="end" font-size="11" fill="#63746a">${date(rows.at(-1).t)} UTC</text></svg>`;
}
function render() {
  $('results').hidden=!run;
  if (!run) return;
  const scores=filteredRows(run.result.scores), visible=filteredRows(records);
  const from=Date.parse(visible[0].t), to=Date.parse(visible.at(-1).t);
  const periodEvents=events.filter(e=>Date.parse(e.endTime)>=from && Date.parse(e.time)<=to);
  $('scored').textContent=`${scores.length.toLocaleString()} / ${visible.length.toLocaleString()}`;
  $('unscored').textContent=(visible.length-scores.length).toLocaleString();
  $('episodes').textContent=periodEvents.length;
  $('pending').textContent=`${periodEvents.filter(e=>(latestReview(e)?.action || 'open')==='open').length} awaiting review`;
  $('score-chart').innerHTML=chart(scores,'anomaly_score',run.result.model.threshold);
  $('signal-chart').innerHTML=chart(visible,$('channel').value);
  const list=periodEvents.filter(e=>$('filter').value==='all' || (latestReview(e)?.action || 'open')==='open');
  page=Math.min(page,Math.max(0,Math.ceil(list.length/20)-1));
  $('events').replaceChildren();
  for(const event of list.slice(page*20,page*20+20)) {
    const row=document.createElement('tr');
    const action=latestReview(event)?.action || 'open';
    const text=[date(event.time),`${Math.round((Date.parse(event.endTime)-Date.parse(event.time))/60000)} min · ${event.count} reading${event.count===1?'':'s'}`,event.channel,action==='open'?'Awaiting review':action==='dismissed'?'Dismissed':'Acknowledged'];
    for(const value of text) { const cell=document.createElement('td'); cell.textContent=value; row.append(cell); }
    const cell=document.createElement('td'), button=document.createElement('button');
    button.className='quiet';button.textContent='Review';button.onclick=()=>detail(event);cell.append(button);row.append(cell);$('events').append(row);
  }
  $('empty').textContent=list.length?'':periodEvents.length?'No episodes awaiting review.':'No flagged episodes in this period.';
  $('page').textContent=list.length?`${page*20+1}–${Math.min((page+1)*20,list.length)} of ${list.length}`:'0 episodes';
  $('previous').disabled=page===0;$('next').disabled=(page+1)*20>=list.length;
  $('model-details').textContent=`Model: ${run.result.model.model_version}. Threshold: ${number(run.result.model.threshold)}. Saved: ${date(run.created)} UTC. Assessment ID: ${run.id}.`;
}
function detail(event) {
  selected=event;$('detail').hidden=false;$('detail-title').textContent=`Episode · ${date(event.time)} UTC`;
  $('explanation').textContent=`${event.channel} contributed the largest reconstruction error at the peak. Score ${number(event.peak)} exceeded threshold ${number(run.result.model.threshold)}. This identifies an unusual pattern, not a faulty component.`;
  $('notes').value='';$('decision').value='acknowledged';$('review-message').textContent='';
  $('review-history').replaceChildren();
  const history=run.reviews.filter(r=>r.episode===event.time);
  for(const review of history) { const p=document.createElement('p');p.textContent=`${date(review.created)} UTC · ${review.action}${review.comment?' — '+review.comment:''}`;$('review-history').append(p); }
  if(!history.length)$('review-history').textContent='No responses yet.';
  $('detail').focus();
}
function loadRun(value) {
  run=value;records=value.records;events=scoreEvents(value.result,'unassigned');page=0;selected=null;$('detail').hidden=true;
  $('dataset').textContent=value.name;$('data-summary').textContent=`${records.length.toLocaleString()} readings · ${date(records[0].t)} to ${date(records.at(-1).t)} UTC`;
  $('history').value=value.id;render();lock(false);
}
async function history() {
  const {runs}=await api('/api/runs');
  $('history').replaceChildren(new Option('Select an assessment',''));
  for(const r of runs)$('history').add(new Option(`${r.name} · ${date(r.created)}`,r.id));
  return runs;
}
$('file').onchange=async()=> {
  const file=$('file').files[0];if(!file || busy)return;
  try {
    if(file.size>25*1024*1024)throw new Error('Choose a CSV smaller than 25 MB.');
    const imported=parseTelemetry(await file.text());
    records=imported;run=null;events=[];selected=null;$('detail').hidden=true;$('history').value='';
    $('dataset').textContent=file.name;$('data-summary').textContent=`${records.length.toLocaleString()} readings · ${date(records[0].t)} to ${date(records.at(-1).t)} UTC`;
    render();lock(false);status('Ready to assess. Your saved assessments remain in history.');
  }catch(error){status(error.message,true);}finally{$('file').value='';}
};
$('assess').onclick=async()=> {
  lock(true);status('Assessing telemetry on this computer…');
  try {const value=await api('/api/assess',{name:$('dataset').textContent,records});await history();loadRun(value);status('Assessment saved. Review the flagged episodes below.');}
  catch(error){status(error.message,true);}finally{lock(false);}
};
$('history').onchange=async()=>{if(!$('history').value)return;lock(true);try{loadRun(await api('/api/run?id='+encodeURIComponent($('history').value)));status('Saved assessment opened.');}catch(error){status(error.message,true);}finally{lock(false);}};
$('review-form').onsubmit=async event=>{
  event.preventDefault();if(!selected || !run)return;
  $('save-review').disabled=true;
  try {await api('/api/review',{run_id:run.id,episode:selected.time,action:$('decision').value,comment:$('notes').value});run=await api('/api/run?id='+run.id);render();detail(selected);$('review-message').textContent='Response saved. Model results are unchanged.';}
  catch(error){$('review-message').textContent=error.message;}finally{$('save-review').disabled=false;}
};
$('close-detail').onclick=()=>{$('detail').hidden=true;selected=null;};
for(const id of ['period','channel','filter'])$(id).onchange=()=>{page=0;render();};
$('previous').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};
$('logout').onclick=async()=>{try{await api('/api/auth/logout',{});location.assign('/auth.html');}catch(error){status(error.message,true);}};
try {
  const {user}=await api('/api/auth/me');if(!user?.username)location.assign('/auth.html');
  $('account').textContent=user.name || 'Account';
  model=await api('/api/model');$('connection').textContent='Model ready';
  const runs=await history();if(runs.length){loadRun(await api('/api/run?id='+runs[0].id));status('Your latest saved assessment.');}
  lock(false);
}catch(error){$('connection').textContent='Connection unavailable';status(error.message,true);}
