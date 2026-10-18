import { CHANNELS, parseTelemetry, windowRows, toCSV } from './data.js';
import { DEMO_UNITS, unitTelemetry, demoAlerts, unitStatus, eventsInWindow } from './units.js';

const $ = selector => document.querySelector(selector);
let selectedUnit = 'unit-01';
let units = DEMO_UNITS;
let records = unitTelemetry(selectedUnit);
let source = 'demo';
let channel = 'V1';
let toastTimer;
let alerts = demoAlerts();
const descriptions = {
  overview: ['Equipment overview', 'See which units need attention, then review the affected signals.'],
  telemetry: ['Signal readings', 'Readings for the selected equipment. Select a different unit on Equipment overview.'],
  alerts: ['Flagged events', 'See which unit and signal was flagged, and record that you reviewed it.'],
  models: ['Model workspace', 'Track the research baseline and the next steps toward validation.'],
};
const time = value => new Date(value).toISOString().slice(11, 16);
const date = value => new Date(value).toISOString().slice(0, 10);
const number = value => new Intl.NumberFormat('en', { maximumFractionDigits: 3 }).format(value);
const currentRows = () => windowRows(records, $('#time-range').value);
const activeEvents = () => source === 'demo' ? eventsInWindow(alerts, currentRows()) : [];
const unitName = id => units.find(unit => unit.id === id)?.name || 'Unassigned equipment';

function selectUnit(id, signal) {
  selectedUnit = id;
  if (source === 'demo') records = unitTelemetry(id);
  channel = signal || 'V1';
  renderTelemetry(); renderAlerts();
}

function renderUnits() {
  const events = activeEvents();
  const evaluated = source === 'demo';
  $('#unit-source-label').textContent = evaluated ? 'DEMO EQUIPMENT' : 'EQUIPMENT NOT IDENTIFIED';
  $('#assessment-source').textContent = evaluated ? 'Demo events' : 'Not assessed';
  $('#unit-cards').innerHTML = units.map(unit => {
    const status = unitStatus(unit.id, events, evaluated);
    const flagged = events.filter(event => event.unitId === unit.id);
    const unreviewed = flagged.filter(event => !event.acknowledged).length;
    return `<button class="unit-card ${status.kind}" data-unit="${unit.id}" aria-pressed="${unit.id === selectedUnit}"><span class="unit-card-top"><span class="equipment-icon" aria-hidden="true">▥</span><span class="unit-status ${status.kind}">${status.label}</span></span><strong>${unit.name}</strong><span class="unit-location">${unit.location}</span><span class="unit-card-description">${flagged.length ? `Flagged signal: ${[...new Set(flagged.map(event => event.channel))].join(', ')} · ${unreviewed} awaiting review` : evaluated ? 'No flagged signals in this period' : 'Import has no equipment ID or model results'}</span><span class="unit-card-action">${unit.id === selectedUnit ? 'Selected equipment' : 'View unit details →'}</span></button>`;
  }).join('');
  const status = unitStatus(selectedUnit, events, evaluated);
  const unitEvents = events.filter(event => event.unitId === selectedUnit);
  $('#unit-detail-title').textContent = unitName(selectedUnit);
  $('#unit-detail-status').className = `unit-status ${status.kind}`;
  $('#unit-detail-status').textContent = status.label;
  $('#unit-detail-body').innerHTML = unitEvents.length ? unitEvents.map(event => `<div class="event-explanation"><div><h3>${event.title}</h3><p>${event.explanation}</p><span class="event-meta">Flagged ${date(event.time)} at ${time(event.time)} UTC · ${event.channel} · Demo event</span></div><div class="event-actions"><button class="button secondary" data-inspect="${event.id}">View ${event.channel} readings</button><button class="button primary" data-ack="${event.id}" ${event.acknowledged ? 'disabled' : ''}>${event.acknowledged ? 'Reviewed' : 'Mark reviewed'}</button></div></div>`).join('') : `<div class="event-explanation"><div><h3>${evaluated ? 'No flagged events in the selected period' : 'Equipment and anomaly status are unknown'}</h3><p>${status.summary} ${evaluated ? 'Select another unit or expand the review period to explore other events.' : 'This CSV contains signals only. A verified equipment assignment and model inference are needed to identify affected units.'}</p></div></div>`;
}

function toast(message) {
  $('#toast').textContent = message;
  $('#toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 5500);
}

function renderRoute() {
  const requested = location.hash.slice(1) || 'overview';
  const view = Object.hasOwn(descriptions, requested) ? requested : 'overview';
  document.querySelectorAll('.view').forEach(element => { element.hidden = element.id !== `view-${view}`; });
  document.querySelectorAll('nav a').forEach(element => {
    element.classList.toggle('active', element.dataset.view === view);
    if (element.dataset.view === view) element.setAttribute('aria-current', 'page');
    else element.removeAttribute('aria-current');
  });
  $('#page-title').textContent = descriptions[view][0];
  $('#page-description').textContent = descriptions[view][1];
  $('#breadcrumb-view').textContent = descriptions[view][0];
  document.title = `HydroPredict · ${descriptions[view][0]}`;
}

function chartMarkup(rows, selected, id) {
  const width = 760, height = 245, left = 49, right = 16, top = 12, bottom = 31;
  let min = Infinity, max = -Infinity;
  for (const row of rows) { min = Math.min(min, row[selected]); max = Math.max(max, row[selected]); }
  const span = max - min || Math.abs(max) * 0.1 || 1;
  min -= span * 0.18; max += span * 0.18;
  // Reject pathological numeric ranges rather than emitting invalid SVG coordinates.
  if (!Number.isFinite(max - min)) return '<div class="empty-state">Numeric range too large to plot. Inspect the source CSV.</div>';
  const start = Date.parse(rows[0].t), end = Date.parse(rows.at(-1).t);
  const x = row => left + (Date.parse(row.t) - start) / (end - start || 1) * (width - left - right);
  const y = row => top + (max - row[selected]) / (max - min) * (height - top - bottom);
  // Bucket extrema preserve narrow spikes without rendering every imported sample.
  const buckets = new Map();
  rows.forEach((row, i) => {
    const key = Math.floor(x(row));
    const bucket = buckets.get(key) || [];
    bucket.push(i); buckets.set(key, bucket);
  });
  const retained = new Set([0, rows.length - 1]);
  buckets.forEach(indices => {
    let low = indices[0], high = indices[0];
    indices.forEach(i => { if (rows[i][selected] < rows[low][selected]) low = i; if (rows[i][selected] > rows[high][selected]) high = i; });
    [indices[0], low, high, indices.at(-1)].forEach(i => retained.add(i));
  });
  rows.forEach((row, i) => { if (i && Date.parse(row.t) - Date.parse(rows[i - 1].t) > 600000) { retained.add(i - 1); retained.add(i); } });
  let path = '';
  [...retained].sort((a, b) => a - b).forEach((i, n) => {
    // Only actual gaps break paths; decimation can skip many continuous samples.
    const actualGap = n === 0 || (i > 0 && Date.parse(rows[i].t) - Date.parse(rows[i - 1].t) > 600000);
    path += `${actualGap ? 'M' : 'L'}${x(rows[i]).toFixed(2)},${y(rows[i]).toFixed(2)} `;
  });
  let grid = '';
  for (let tick = 0; tick < 5; tick++) {
    const yy = top + tick / 4 * (height - top - bottom);
    grid += `<line x1="${left}" x2="${width - right}" y1="${yy}" y2="${yy}" stroke="#edf1ec" stroke-dasharray="3 5"/><text x="${left - 10}" y="${yy + 3}" text-anchor="end">${number(max - tick / 4 * (max - min))}</text>`;
  }
  for (let tick = 0; tick < 6; tick++) {
    const xx = left + tick / 5 * (width - left - right);
    grid += `<text x="${xx}" y="${height - 7}" text-anchor="middle">${time(start + tick / 5 * (end - start))}</text>`;
  }
  const markers = activeEvents().filter(event => event.unitId === selectedUnit && event.channel === selected).map(event => {
    const xx = x({ t: event.time });
    return `<line x1="${xx}" x2="${xx}" y1="${top}" y2="${height - bottom}" stroke="#ba7041" stroke-dasharray="4 4"/><text x="${Math.min(Math.max(xx, 110), width - 80)}" y="${top + 10}" text-anchor="middle" class="event-chart-label">Flagged ${time(event.time)}</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${id}-title"><title id="${id}-title">${unitName(selectedUnit)}, ${selected} source measurements from ${rows[0].t} to ${rows.at(-1).t}. Values range from ${number(min)} to ${number(max)}. Measurement units unverified. Dashed orange lines mark demo events.</title>${grid}<path d="${path}" fill="none" stroke="#359779" stroke-width="2" stroke-linejoin="round"/>${markers}<circle cx="${x(rows.at(-1))}" cy="${y(rows.at(-1))}" r="3" fill="#359779"/></svg>`;
}

function renderTelemetry() {
  const rows = currentRows();
  $('#observation-count').textContent = rows.length.toLocaleString();
  document.querySelectorAll('.channel-tabs').forEach(element => {
    element.innerHTML = CHANNELS.map(c => `<button aria-pressed="${c === channel}" data-channel="${c}">${c}</button>`).join('');
  });
  document.querySelectorAll('.selected-channel').forEach(element => { element.textContent = `${unitName(selectedUnit)} / ${channel}`; });
  $('#overview-chart').innerHTML = chartMarkup(rows, channel, 'overview');
  $('#explorer-chart').innerHTML = chartMarkup(rows, channel, 'explorer');
  $('#chart-range').textContent = `${date(rows[0].t)} — ${date(rows.at(-1).t)}`;
  $('#channel-snapshot').innerHTML = CHANNELS.map(c => {
    const recent = rows.slice(-24);
    const values = recent.map(row => row[c]);
    const min = Math.min(...values), span = Math.max(...values) - min || 1;
    const points = values.map((value, i) => `${i / Math.max(values.length - 1, 1) * 66},${18 - (value - min) / span * 15}`).join(' ');
    const sparkline = Number.isFinite(span) ? `<svg viewBox="0 0 66 20" aria-hidden="true"><polyline points="${points}" fill="none" stroke="#87b49a" stroke-width="1.3"/></svg>` : '';
    return `<div class="channel-row"><div class="channel-name"><span></span>${c}</div>${sparkline}<div class="channel-number">${number(rows.at(-1)[c])}</div></div>`;
  }).join('');
  $('#telemetry-table').innerHTML = `<table><thead><tr><th>TIMESTAMP (UTC)</th>${CHANNELS.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.slice(-20).reverse().map(row => `<tr><td>${date(row.t)} ${time(row.t)}</td>${CHANNELS.map(c => `<td>${number(row[c])}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

function renderAlerts() {
  const visible = activeEvents();
  const selected = visible.filter(event => event.unitId === selectedUnit);
  $('#open-alert-count').textContent = source === 'demo' ? selected.filter(event => !event.acknowledged).length : '—';
  $('#nav-alert-count').textContent = visible.filter(event => !event.acknowledged).length;
  $('#recent-count').textContent = selected.length;
  $('#alert-metric-description').textContent = source === 'demo' ? `${unitName(selectedUnit)} · demo events` : 'No model connected · status unknown';
  const table = items => !items.length ? `<div class="empty-state">${source === 'demo' ? 'No flagged events match this equipment, period, and review status.' : 'Anomaly status unknown. No model has assessed this import.'}</div>` : `<div class="table-scroll"><table><thead><tr><th>EQUIPMENT / SIGNAL</th><th>WHAT CHANGED</th><th>FLAGGED AT (UTC)</th><th>REVIEW STATUS</th><th>ACTION</th></tr></thead><tbody>${items.map(event => `<tr><td><strong>${unitName(event.unitId)}</strong><br><small>Signal ${event.channel}</small></td><td><strong>${event.title}</strong><br><small>${event.id} · Demo event</small></td><td>${time(event.time)}<br><small>${date(event.time)}</small></td><td><span class="alert-status ${event.acknowledged ? 'done' : ''}">${event.acknowledged ? 'Reviewed · still flagged' : 'Awaiting review'}</span></td><td><button class="ack-button" data-inspect="${event.id}">View unit</button></td></tr>`).join('')}</tbody></table></div>`;
  $('#recent-alerts').innerHTML = table(selected);
  const filter = $('#alert-filter').value;
  $('#all-alerts').innerHTML = table(visible.filter(event => filter === 'all' || (filter === 'acknowledged' ? event.acknowledged : !event.acknowledged)));
  renderUnits();
}

document.addEventListener('click', event => {
  const unit = event.target.closest('[data-unit]');
  if (unit) {
    selectUnit(unit.dataset.unit);
    document.querySelector(`[data-unit="${selectedUnit}"]`).focus({ preventScroll: true });
  }
  const inspect = event.target.closest('[data-inspect]');
  if (inspect) {
    const flagged = alerts.find(item => item.id === inspect.dataset.inspect);
    selectUnit(flagged.unitId, flagged.channel);
    location.hash = 'overview';
    renderRoute();
    $('#unit-detail-title').scrollIntoView({ behavior: 'smooth', block: 'center' });
    const tab = document.querySelector(`#overview-channel-tabs [data-channel="${flagged.channel}"]`);
    tab.focus({ preventScroll: true });
    toast(`Showing ${unitName(flagged.unitId)}, signal ${flagged.channel}. Orange dashed line marks the demo event.`);
  }
  const selected = event.target.closest('[data-channel]');
  if (selected) { channel = selected.dataset.channel; renderTelemetry(); }
  const acknowledgement = event.target.closest('[data-ack]');
  if (acknowledgement) {
    alerts.find(alert => alert.id === acknowledgement.dataset.ack).acknowledged = true;
    renderAlerts(); toast('Marked reviewed. The anomaly stays flagged; this does not mark equipment repaired.');
  }
});
$('#time-range').addEventListener('change', () => { renderTelemetry(); renderAlerts(); });
$('#alert-filter').addEventListener('change', renderAlerts);
$('#import-button').addEventListener('click', () => $('#csv-file').click());
$('#csv-file').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    if (file.size > 25 * 1024 * 1024) throw new Error('Choose a CSV smaller than 25 MB.');
    const imported = parseTelemetry(await file.text());
    records = imported; source = 'import';
    selectedUnit = 'unassigned';
    units = [{ id: 'unassigned', name: 'Unassigned equipment', location: 'Imported CSV · equipment identity unknown' }];
    channel = 'V1';
    $('#source-pill').textContent = 'Imported CSV';
    $('#dataset-label').textContent = file.name;
    $('#notice-text').textContent = 'Imported source measurements. No inference is running. Naive timestamps are interpreted as UTC; physical units remain unverified.';
    $('#time-range').value = 'all';
    renderTelemetry(); renderAlerts();
    toast(`Imported ${records.length.toLocaleString()} observations. Data stays in this tab.`);
  } catch (error) { toast(`Import failed: ${error.message}`); }
  finally { event.target.value = ''; }
});
$('#reset-demo').addEventListener('click', () => {
  source = 'demo'; units = DEMO_UNITS; selectedUnit = 'unit-01';
  records = unitTelemetry(selectedUnit); alerts = demoAlerts(); channel = 'V1';
  $('#source-pill').textContent = 'Demo dataset';
  $('#dataset-label').textContent = 'SHP / Demo telemetry';
  $('#notice-text').textContent = 'Three example units with synthetic readings and demo events. Equipment assignments are illustrative; no live model is connected.';
  $('#time-range').value = '24'; $('#alert-filter').value = 'all';
  renderTelemetry(); renderAlerts(); toast('Demo dataset restored.');
});
$('#export-button').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([toCSV(currentRows())], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url; link.download = `hydropredict-${source}-${selectedUnit}-window.csv`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast('Exported all six channels for the selected time window.');
});
$('.mini-bars').innerHTML = Array.from({ length: 26 }, (_, i) => `<i style="height:${9 + Math.round((Math.sin(i * 1.8) + 1) * 5)}px"></i>`).join('');
window.addEventListener('hashchange', renderRoute);
renderRoute(); renderTelemetry(); renderAlerts();
