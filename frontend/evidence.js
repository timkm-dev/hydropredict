import { LABELS, scoreSegments, describeEvent, validateEvidence } from './evidence-model.js';

const $ = id => document.getElementById(id);
const utc = value => new Date(value).toISOString().replace('T', ' ').slice(0, 19);
const date = value => new Date(value).toISOString().slice(0, 10);
const pct = value => `${(value * 100).toFixed(1)}%`;
const escape = text => String(text).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
const lead = value => value == null ? '—' : value >= 60 ? `${Math.floor(value / 60)}h ${Math.round(value % 60)}m` : `${value.toFixed(1)} min`;
let evidence;
let selected = 0;

function overview() {
  const width = 1100, height = 155, left = 88, right = 25;
  const start = Date.parse(evidence.period[0]), stop = Date.parse(evidence.period[1]);
  const x = t => left + (Date.parse(t) - start) / (stop - start) * (width - left - right);
  const anomalies = evidence.scores.filter(row => row.alert);
  $('overview').innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="overview-title overview-desc"><title id="overview-title">Anomalies and recorded faults, February to April 2019</title><desc id="overview-desc">${anomalies.length} anomalous readings and ${evidence.events.length} recorded fault timestamps. Use the numbered buttons below to inspect a fault.</desc>
    <text x="0" y="45">Anomalies</text><text x="0" y="92">Faults</text><line x1="${left}" y1="42" x2="${width - right}" y2="42" stroke="#e5ece7"/><line x1="${left}" y1="88" x2="${width - right}" y2="88" stroke="#e5ece7"/>
    ${anomalies.map(row => `<line x1="${x(row.t)}" y1="32" x2="${x(row.t)}" y2="52" stroke="#c47a33" stroke-width="2"><title>Anomaly ${utc(row.t)} UTC · ${row.ratio.toFixed(2)} × threshold</title></line>`).join('')}
    ${evidence.events.map(event => `<circle cx="${x(event.fault)}" cy="88" r="5" fill="#bb5555"><title>Fault ${event.id} · ${utc(event.fault)} UTC · ${LABELS[event.status]}</title></circle>`).join('')}
    ${['2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01'].map(t => `<line x1="${x(t)}" y1="22" x2="${x(t)}" y2="106" stroke="#dfe8e2" stroke-dasharray="3 5"/><text x="${x(t)}" y="132" text-anchor="middle">${t.slice(5)}</text>`).join('')}</svg>`;
  $('fault-buttons').innerHTML = evidence.events.map((event, index) => `<button class="${event.status}" data-event="${index}" aria-pressed="false"><span class="status-marker"></span>Fault ${event.id} · ${date(event.fault).slice(5)}</button>`).join('');
  $('association').textContent = `${evidence.summary.alert_points} anomaly readings form ${evidence.summary.episodes} episodes. ${evidence.summary.associated_alert_points} reading falls inside a recorded fault’s six-hour warning window; ${evidence.summary.outside_alert_points} fall outside all such windows. Outside-window anomalies are not automatically false alarms: fault records may be incomplete and unusual behavior can have other causes.`;
}

function eventChart(event) {
  const history = Number($('history').value);
  const fault = Date.parse(event.fault), start = fault - history * 3600000, stop = fault + 1800000;
  const rows = evidence.scores.filter(row => Date.parse(row.t) >= start && Date.parse(row.t) <= stop);
  const width = 820, height = 310, left = 48, right = 18, top = 25, bottom = 43;
  const ymax = Math.max(1.3, ...rows.map(row => row.ratio * 1.12));
  const x = t => left + (t - start) / (stop - start) * (width - left - right);
  const y = ratio => height - bottom - ratio / ymax * (height - top - bottom);
  const segments = scoreSegments(rows);
  const missing = [];
  let cursor = start;
  for (const segment of segments) {
    const begin = Date.parse(segment[0].t), end = Date.parse(segment.at(-1).t);
    if (begin - cursor > 450000) missing.push([cursor, begin]);
    cursor = end;
  }
  if (stop - cursor > 450000) missing.push([cursor, stop]);
  const ticks = history === 6 ? [-6, -4, -2, 0] : [-12, -9, -6, -3, 0];
  $('event-chart').innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="event-chart-title event-chart-description"><title id="event-chart-title">Anomaly scores around fault ${event.id}</title><desc id="event-chart-description">${LABELS[event.status]}. ${event.scored_points} scored readings in the six-hour warning window; ${event.alert_points} exceed the threshold. Gray bands indicate unavailable model scores.</desc>
    <rect x="${x(fault - 21600000)}" y="${top}" width="${x(fault) - x(fault - 21600000)}" height="${height - top - bottom}" fill="#eaf5ef"/>
    ${missing.map(([a, b]) => `<rect x="${x(a)}" y="${top}" width="${x(b) - x(a)}" height="${height - top - bottom}" fill="#e6eae7" fill-opacity=".8"/>`).join('')}
    ${[0, 0.5, 1, ...(ymax > 2 ? [Math.floor(ymax)] : [])].map(v => `<line x1="${left}" y1="${y(v)}" x2="${width - right}" y2="${y(v)}" stroke="${v === 1 ? '#bc792f' : '#dbe5de'}" stroke-dasharray="${v === 1 ? '6 4' : '0'}"/><text x="${left - 9}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(1)}</text>`).join('')}
    ${ticks.map(hour => `<text x="${x(fault + hour * 3600000)}" y="${height - 14}" text-anchor="middle">${hour === 0 ? 'Fault' : `${hour}h`}</text>`).join('')}
    ${segments.filter(s => s.length > 1).map(segment => `<polyline points="${segment.map(row => `${x(Date.parse(row.t))},${y(row.ratio)}`).join(' ')}" fill="none" stroke="#167d70" stroke-width="1.8"/>`).join('')}
    ${rows.map(row => `<circle cx="${x(Date.parse(row.t))}" cy="${y(row.ratio)}" r="${row.alert ? 4 : 2}" class="${row.alert ? 'alert' : 'point'}"><title>${utc(row.t)} UTC · ${row.ratio.toFixed(3)} × threshold${row.alert ? ' · anomaly' : ''}</title></circle>`).join('')}
    <line x1="${x(fault)}" y1="${top - 5}" x2="${x(fault)}" y2="${height - bottom}" stroke="#bc5353" stroke-width="2" stroke-dasharray="5 4"/>
    <text x="${left}" y="14">Score / frozen threshold</text>${!rows.length ? `<text x="${width / 2}" y="${height / 2}" text-anchor="middle">No eligible model scores in this period</text>` : ''}</svg>`;
}

function selectEvent(index) {
  selected = Math.max(0, Math.min(evidence.events.length - 1, index));
  const event = evidence.events[selected];
  const explanation = describeEvent(event);
  $('event-title').textContent = `Fault ${event.id} · ${LABELS[event.status]}`;
  $('event-date').textContent = `${utc(event.fault)} UTC · fault group ${event.group}`;
  $('event-explanation').innerHTML = `<span class="badge ${event.status}">${LABELS[event.status]}</span><h3>${explanation.title}</h3><p>${explanation.body}</p><dl><div><dt>Six-hour score coverage</dt><dd>${pct(event.coverage)}</dd></div><div><dt>Peak / threshold</dt><dd>${event.peak_ratio == null ? 'Unavailable' : `${event.peak_ratio.toFixed(2)} ×`}</dd></div><div><dt>First warning lead</dt><dd>${lead(event.lead_minutes)}</dd></div><div><dt>Alert readings</dt><dd>${event.alert_points}</dd></div><div><dt>Latest score before fault</dt><dd>${event.latest_gap_minutes == null ? 'Unavailable' : `${event.latest_gap_minutes.toFixed(1)} min`}</dd></div></dl>`;
  $('previous').disabled = selected === 0;
  $('next').disabled = selected === evidence.events.length - 1;
  document.querySelectorAll('[data-event]').forEach(button => button.setAttribute('aria-pressed', Number(button.dataset.event) === selected));
  document.querySelectorAll('#event-table tr').forEach((row, i) => row.classList.toggle('selected', i === selected));
  history.replaceState(null, '', `#fault-${event.id}`);
  eventChart(event);
}

async function initialize() {
  try {
    const response = await fetch('./evidence-data.json', { cache: 'no-store' });
    if (!response.ok) throw new Error('The local evidence file has not been generated.');
    evidence = validateEvidence(await response.json());
    const s = evidence.summary;
    $('stats').innerHTML = [
      ['', 'Recorded faults', s.faults, 'Every timestamp stays in the comparison'],
      ['warned', 'Warning observed', `${s.warned}<small> / ${s.eligible} assessable</small>`, 'An anomaly appeared in the preceding 6h'],
      ['missed', 'Missed warnings', s.missed, 'Enough scored history, no anomaly warning'],
      ['insufficient', 'Insufficient data', s.insufficient, 'Not counted as detections or misses'],
    ].map(([kind, title, value, detail]) => `<article class="stat ${kind}"><span>${title}</span><strong>${value}</strong><p>${detail}</p></article>`).join('');
    $('event-table').innerHTML = evidence.events.map((event, index) => `<tr><td><button data-event="${index}" aria-label="Inspect fault ${event.id}">${event.id}. ${utc(event.fault)}</button></td><td><span class="badge ${event.status}">${LABELS[event.status]}</span></td><td>${pct(event.coverage)}</td><td>${event.alert_points}</td><td>${lead(event.lead_minutes)}</td></tr>`).join('');
    $('source').textContent = `${evidence.model} · frozen threshold ${evidence.threshold.toPrecision(7)} · ${s.score_points.toLocaleString()} scores from ${s.observations.toLocaleString()} source observations. On ${s.nominal_points.toLocaleString()} selected nominal-reference scores, ${s.nominal_alert_points} were flagged (${pct(s.nominal_alert_points / s.nominal_points)}). This percentage is not an operational false-alarm guarantee.`;
    $('hashes').textContent = JSON.stringify(evidence.provenance, null, 2);
    overview();
    $('evidence').hidden = false;
    $('load-status').hidden = true;
    const id = Number(location.hash.match(/^#fault-(\d+)$/)?.[1]);
    selectEvent(Math.max(0, evidence.events.findIndex(event => event.id === id)));
  } catch (error) {
    $('load-status').innerHTML = `<strong>Research evidence is unavailable.</strong><p>${escape(error.message)}</p><p>From the repository root, generate the local evidence and reload this page:</p><code>.venv/Scripts/python.exe scripts/map_anomalies_to_faults.py</code><p>The original local input package and frozen diagnostic results are required. No example results are substituted.</p>`;
  }
}

document.addEventListener('click', event => {
  const button = event.target.closest('[data-event]');
  if (button && evidence) selectEvent(Number(button.dataset.event));
});
$('previous').onclick = () => selectEvent(selected - 1);
$('next').onclick = () => selectEvent(selected + 1);
$('history').onchange = () => eventChart(evidence.events[selected]);
window.addEventListener('hashchange', () => {
  if (!evidence) return;
  const id = Number(location.hash.match(/^#fault-(\d+)$/)?.[1]);
  const index = evidence.events.findIndex(event => event.id === id);
  if (index >= 0) selectEvent(index);
});
initialize();
