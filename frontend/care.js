import {
  ACTION_LABEL, STATUS_LABEL, WATCH, awaitingReview, fleetCounts, formatHours, formatTime,
  indexAt, latestReview, linePath, statusOf, upperBound,
} from './care-model.js';

const $ = id => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
let fleet = null, reviews = [], period = null, cursor = 0, timer = null, sensor = null;
const cache = new Map();

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === 'class') node.className = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (key in node && key !== 'type') node[key] = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.filter(c => c !== null && c !== undefined && c !== false));
  return node;
}
function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
function pill(status, label = STATUS_LABEL[status]) { return el('span', { class: `pill ${status}` }, label); }

async function api(path, body) {
  const response = await fetch(path, body === undefined
    ? { cache: 'no-store' }
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (response.status === 401) { location.assign('/auth.html'); throw new Error('Please sign in.'); }
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed. Try again.');
  return result;
}
async function periodData(key) {
  if (!cache.has(key)) cache.set(key, api(`care-data/${key}.json`));
  return cache.get(key);
}
function allAlerts() {
  return fleet.periods.flatMap(p => p.alerts.map(a => ({ ...a, period: p })));
}
function updateCount() {
  const waiting = allAlerts().filter(a => awaitingReview(reviews, a.id)).length;
  $('nav-count').hidden = !waiting;
  $('nav-count').textContent = waiting;
}

// ---------- Fleet ----------
function renderFleet() {
  const counts = fleetCounts(fleet.periods, reviews);
  for (const key of ['alarm', 'watch', 'normal', 'awaiting']) $(`count-${key}`).textContent = counts[key];
  const farm = $('filter-farm').value, status = $('filter-status').value;
  const rows = fleet.periods.filter(p => (!farm || p.farm === farm) && (!status || statusOf(p.max_crit) === status));
  $('fleet-rows').replaceChildren(...rows.map(p => {
    const first = p.alerts[0];
    const row = el('tr', { class: 'link', tabIndex: 0, onclick: () => { location.hash = `#/period/${p.key}`; },
      onkeydown: e => { if (e.key === 'Enter') location.hash = `#/period/${p.key}`; } },
      el('td', {}, el('strong', {}, p.key), el('small', {}, `${p.farm} · turbine ${p.asset}`)),
      el('td', {}, pill(statusOf(p.max_crit))),
      el('td', {}, p.look_first || '—'),
      el('td', {}, first ? formatTime(first.start) : '—'),
      el('td', {}, String(p.alerts.length)),
      el('td', {}, p.recorded_fault ? el('span', { title: p.recorded_fault.description }, shorten(p.recorded_fault.description)) : 'None recorded'));
    return row;
  }));
}
function shorten(text, n = 48) { return text.length > n ? `${text.slice(0, n - 1)}…` : text; }

// ---------- Period ----------
async function showPeriod(key) {
  stop();
  const summary = fleet.periods.find(p => p.key === key);
  if (!summary) { message('That monitoring period is not in the replay data.', true); location.hash = '#/'; return; }
  message('Loading period…');
  period = await periodData(key);
  period.times = period.hours.map(Date.parse);
  message('');
  $('period-eyebrow').textContent = `${period.farm.toUpperCase()} · TURBINE ${period.asset}`;
  $('period-title').textContent = `Monitoring period ${period.key}`;
  const fault = period.fault;
  $('period-summary').textContent = fault
    ? `Recorded fault: ${fault.description} (period ${formatTime(fault.start)} to ${formatTime(fault.end)}).`
    : 'No fault recorded in this period.';
  const keys = Object.keys(period.series);
  $('sensor').replaceChildren(...keys.map(i => el('option', { value: i }, period.sensors[i].description)));
  sensor = keys[0] ?? null;
  $('cursor').max = period.hours.length - 1;
  setCursor(period.hours.length - 1);
  renderPeriodAlerts();
}
function setCursor(index) {
  cursor = Math.max(0, Math.min(index, period.hours.length - 1));
  $('cursor').value = cursor;
  const crit = period.crit[cursor], status = statusOf(crit, period.alarm_level);
  $('clock').textContent = `Showing ${formatTime(period.hours[cursor])} UTC`;
  $('crit-value').textContent = `${Math.min(crit, period.alarm_level)} / ${period.alarm_level}`;
  $('crit-bar').style.width = `${Math.min(100, crit / period.alarm_level * 100)}%`;
  $('crit-bar').className = status;
  $('period-status').replaceWith(Object.assign(pill(status), { id: 'period-status' }));
  // Alarms stay visible after the counter falls, as in plant alarm systems, until reviewed.
  const raised = period.alerts.filter(a => Date.parse(a.start) <= period.times[cursor]).at(-1);
  const open = raised && awaitingReview(reviews, raised.id);
  $('period-latch').hidden = !raised || status === 'alarm';
  if (raised) $('period-latch').textContent = `Alarm raised ${formatTime(raised.start)}${open ? ' · awaiting review' : ''}`;
  renderScoreChart();
  renderLookFirst();
  renderSensorChart();
}
function frame(width, height) {
  return svg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img' });
}
function renderScoreChart() {
  const W = 960, H = 230, L = 50, R = 16, T = 14, B = 30;
  const t0 = period.times[0], t1 = period.times.at(-1);
  const high = upperBound(period.score, period.threshold);
  const x = t => L + (t - t0) / (t1 - t0 || 1) * (W - L - R);
  const y = v => T + (1 - Math.min(v, high) / high) * (H - T - B);
  const chart = frame(W, H);
  chart.setAttribute('aria-label', 'Warning score over time with threshold, recorded fault period and alarms');
  if (period.fault) {
    const a = x(Math.max(t0, Date.parse(period.fault.start))), b = x(Math.min(t1, Date.parse(period.fault.end)));
    if (b > a) chart.append(svg('rect', { x: a, y: T, width: b - a, height: H - T - B, fill: '#f3e3c3' }));
  }
  chart.append(svg('line', { x1: L, x2: W - R, y1: y(period.threshold), y2: y(period.threshold), stroke: '#a53328', 'stroke-dasharray': '5 4' }));
  const shown = period.score.map((v, i) => (i <= cursor ? v : null));
  chart.append(svg('path', { d: linePath(period.times, shown, x, y), fill: 'none', stroke: '#205c4b', 'stroke-width': 2 }));
  for (const alert of period.alerts) {
    const t = Date.parse(alert.start);
    if (t <= period.times[cursor]) chart.append(svg('line', { x1: x(t), x2: x(t), y1: T, y2: H - B, stroke: '#a53328', 'stroke-width': 2 }));
  }
  chart.append(svg('line', { x1: x(period.times[cursor]), x2: x(period.times[cursor]), y1: T, y2: H - B, stroke: '#203b37', 'stroke-dasharray': '2 3' }));
  chart.append(svg('text', { x: L, y: H - 8 }, formatTime(period.hours[0]).slice(0, 10)));
  chart.append(svg('text', { x: W - R, y: H - 8, 'text-anchor': 'end' }, formatTime(period.hours.at(-1)).slice(0, 10)));
  $('score-chart').replaceChildren(chart);
}
function renderLookFirst() {
  const top = period.top[cursor] || [];
  if (period.crit[cursor] < WATCH) {
    $('look-first').replaceChildren(el('li', { class: 'caption' }, 'No warning at this time. The list appears when the warning counter reaches watch level; the strongest deviations are still shown below.'));
  } else $('look-first').replaceChildren();
  const max = Math.max(...top.map(([, s]) => s), 0.01);
  $('look-first').append(...top.map(([index, share]) => {
    const info = period.sensors[index];
    const context = info.component === 'Operating conditions';
    const button = el('button', { type: 'button', 'aria-pressed': String(String(index) === sensor),
      onclick: () => { if (period.series[index]) { sensor = String(index); $('sensor').value = sensor; renderSensorChart(); renderLookFirst(); } } },
      el('span', { class: context ? 'context' : '' }, info.description, el('small', {}, context ? 'Operating conditions · context, not component health' : info.component)),
      el('span', { class: 'bar' }, el('i', { style: `width:${share / max * 100}%` })),
      el('span', {}, `${Math.round(share * 100)}%`));
    return el('li', {}, button);
  }));
}
function renderSensorChart() {
  if (sensor === null || !period.series[sensor]) { $('sensor-chart').replaceChildren(el('p', { class: 'caption' }, 'Choose a sensor.')); return; }
  const { actual, expected } = period.series[sensor];
  const W = 640, H = 220, L = 54, R = 12, T = 12, B = 28;
  const values = [...actual, ...expected].filter(Number.isFinite);
  let low = Math.min(...values), high = Math.max(...values);
  const pad = (high - low) * 0.1 || 1; low -= pad; high += pad;
  const t0 = period.times[0], t1 = period.times.at(-1);
  const x = t => L + (t - t0) / (t1 - t0 || 1) * (W - L - R);
  const y = v => T + (high - v) / (high - low) * (H - T - B);
  const chart = frame(W, H);
  chart.setAttribute('aria-label', `${period.sensors[sensor].description}: actual and expected values`);
  const upto = arr => arr.map((v, i) => (i <= cursor ? v : null));
  chart.append(svg('path', { d: linePath(period.times, upto(expected), x, y), fill: 'none', stroke: '#7d8f86', 'stroke-dasharray': '5 4', 'stroke-width': 1.5 }));
  chart.append(svg('path', { d: linePath(period.times, upto(actual), x, y), fill: 'none', stroke: '#205c4b', 'stroke-width': 2 }));
  const unit = period.sensors[sensor].unit;
  chart.append(svg('text', { x: 4, y: T + 10 }, `${high.toPrecision(3)} ${unit}`));
  chart.append(svg('text', { x: 4, y: H - B }, `${low.toPrecision(3)} ${unit}`));
  $('sensor-chart').replaceChildren(chart);
}
function renderPeriodAlerts() {
  if (!period.alerts.length) { $('period-alerts').replaceChildren(el('p', { class: 'caption' }, 'No alarm was raised in this period.')); return; }
  $('period-alerts').replaceChildren(...period.alerts.map(alert => {
    const last = latestReview(reviews, alert.id);
    const lead = period.fault ? (Date.parse(period.fault.end) - Date.parse(alert.start)) / 3600e3 : null;
    return el('div', { class: 'period-alert' },
      el('span', {}, el('strong', {}, formatTime(alert.start)), ` · ${period.sensors[alert.top[0]?.[0]]?.component ?? 'Other'}`,
        lead !== null && lead > 0 ? el('small', {}, ` ${formatHours(lead)} before the recorded fault period ended`) : null),
      el('span', {}, last ? pill('done', ACTION_LABEL[last.action]) : pill('alarm', 'Awaiting review'), ' ',
        el('a', { href: `#/alerts/${alert.id}` }, 'Review')));
  }));
}
function stop() {
  if (timer) clearInterval(timer);
  timer = null;
  $('play-label').textContent = 'Replay history';
}
function togglePlay() {
  if (timer) { stop(); return; }
  if (cursor >= period.hours.length - 1) setCursor(0);
  $('play-label').textContent = 'Pause';
  timer = setInterval(() => {
    const step = Number($('speed').value);
    const next = indexAt(period.times, period.times[cursor] + step * 3600e3);
    setCursor(Math.max(next, cursor + 1));
    if (cursor >= period.hours.length - 1) stop();
  }, 120);
}

// ---------- Alerts ----------
function renderAlerts(focus) {
  const filter = $('alert-filter').value;
  const alerts = allAlerts().filter(a => filter === 'all' || awaitingReview(reviews, a.id) || a.id === focus)
    .sort((a, b) => b.peak - a.peak);
  $('alerts-empty').hidden = alerts.length > 0;
  $('alert-list').replaceChildren(...alerts.map(alert => alertCard(alert)));
  if (focus) document.getElementById(`alert-${focus}`)?.scrollIntoView({ block: 'center' });
}
function alertCard(alert) {
  const last = latestReview(reviews, alert.id);
  const notes = el('textarea', { rows: 2, maxLength: 2000, placeholder: 'What did you see or decide?', 'aria-label': `Notes for ${alert.id}` });
  const error = el('p', { class: 'error', role: 'status' });
  notes.addEventListener('input', () => { error.textContent = ''; });
  const decide = action => async event => {
    event.preventDefault();
    if (action === 'dismissed' && !notes.value.trim()) { error.textContent = 'Add a reason before dismissing.'; notes.focus(); return; }
    try {
      reviews = (await api('/api/care/review', { alert: alert.id, action, comment: notes.value })).reviews;
      updateCount();
      message(`${alert.id}: ${ACTION_LABEL[action].toLowerCase()}.`);
      renderAlerts(alert.id);
    } catch (e) { error.textContent = e.message; }
  };
  const history = reviews.filter(r => r.alert === alert.id);
  const actions = last && last.action !== 'open'
    ? [el('button', { class: 'quiet', onclick: decide('open') }, 'Reopen for review')]
    : [el('button', { onclick: decide('inspect') }, 'Send for inspection'),
      el('button', { class: 'quiet', onclick: decide('acknowledged') }, 'Acknowledge'),
      el('button', { class: 'quiet', onclick: decide('dismissed') }, 'Dismiss')];
  return el('article', { class: 'alert-card', id: `alert-${alert.id}` },
    el('header', {},
      el('div', {}, el('h3', {}, `${alert.period.key} · alert ${alert.id.split('-').at(-1)} · ${alert.component}`),
        el('p', { class: 'caption' }, `Raised ${formatTime(alert.start)} UTC · ${alert.period.farm}, turbine ${alert.period.asset} · ${alert.peak >= 1000 ? 'warning counter reached its maximum' : `warning counter peaked at ${alert.peak}`}`)),
      last ? pill('done', ACTION_LABEL[last.action]) : pill('alarm', 'Awaiting review')),
    el('p', {}, 'Look here first: ', alert.top.join(', ') || '—'),
    el('p', {}, el('a', { href: `#/period/${alert.period.key}` }, 'Open the period replay')),
    el('form', { onsubmit: e => e.preventDefault() }, notes, el('div', { class: 'actions' }, ...actions), error),
    history.length ? el('div', { class: 'history' }, ...history.map(r => el('div', {},
      `${formatTime(r.created)} · ${ACTION_LABEL[r.action]}${r.comment ? ` · “${r.comment}”` : ''}`))) : null);
}

// ---------- Routing ----------
async function route() {
  const [, view = '', id = ''] = location.hash.split('/');
  for (const name of ['fleet', 'period', 'alerts']) $(`view-${name}`).hidden = true;
  for (const link of document.querySelectorAll('[data-nav]')) link.removeAttribute('aria-current');
  try {
    if (view === 'period' && id) { $('view-period').hidden = false; await showPeriod(decodeURIComponent(id)); }
    else if (view === 'alerts') {
      stop();
      $('view-alerts').hidden = false;
      document.querySelector('[data-nav=alerts]').setAttribute('aria-current', 'page');
      renderAlerts(id ? decodeURIComponent(id) : null);
    } else {
      stop();
      $('view-fleet').hidden = false;
      document.querySelector('[data-nav=fleet]').setAttribute('aria-current', 'page');
      renderFleet();
    }
    window.scrollTo(0, 0);
  } catch (error) { message(error.message, true); }
}

$('filter-farm').onchange = renderFleet;
$('filter-status').onchange = renderFleet;
$('alert-filter').onchange = () => renderAlerts();
$('play').onclick = togglePlay;
$('cursor').oninput = () => { stop(); setCursor(Number($('cursor').value)); };
$('sensor').onchange = () => { sensor = $('sensor').value; renderSensorChart(); renderLookFirst(); };
$('logout').onclick = async () => {
  await fetch('/api/auth/logout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  location.replace('/auth.html');
};
window.addEventListener('hashchange', route);

try {
  message('Loading replay data…');
  [fleet, { reviews }] = await Promise.all([api('care-data/fleet.json'), api('/api/care/reviews')]);
  message('');
  updateCount();
  await route();
} catch (error) {
  message(`${error.message} Generate the replay data with scripts/build_care_app_data.py, then reload.`, true);
}
