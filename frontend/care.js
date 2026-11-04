import {
  ACTION_LABEL, DISMISS_REASONS, STATUS_LABEL, WATCH, awaitingReview, cleanUnit, describeDeviation,
  describeLead, fleetCounts, formatDate, formatMoment, formatValue, indexAt, latestReview, linePath,
  statusOf, upperBound, warningLead,
} from './care-model.js';
import { ROLE_BADGE, showForRole } from './accounts.js';

const $ = id => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const SHOWN = 6;
let fleet = null, reviews = [], period = null, cursor = 0, timer = null, sensor = null;
let showAllInspect = false, showAllWatch = false, showAllAlerts = false;
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
function turbineName(p) { return `Turbine ${p.asset} · ${p.farm}`; }
function span(p) { return `${formatDate(p.start)} to ${formatDate(p.end)}`; }
function listText(items) {
  return items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items.at(-1)}`;
}

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

// ---------- Decisions (shared by the turbine page and the alerts inbox) ----------
// Alerts decided since the page loaded offer Undo; older decisions offer Reopen.
const decidedNow = new Set();
function announce(text) { $('announce').textContent = text; }

// `done` redraws the decision; it receives the action so callers can move focus.
function decisionControls(alert, done) {
  const last = latestReview(reviews, alert.id);
  const box = el('div', { class: 'decision' });
  const error = el('p', { class: 'error-text', role: 'alert' });
  const save = async (button, action, comment = '') => {
    const controls = [...box.querySelectorAll('button, select, textarea')];
    const label = button.textContent;
    controls.forEach(c => { c.disabled = true; });
    button.textContent = 'Saving…';
    box.setAttribute('aria-busy', 'true');
    error.textContent = '';
    try {
      reviews = (await api('/api/care/review', { alert: alert.id, action, comment })).reviews;
      if (action === 'open') decidedNow.delete(alert.id); else decidedNow.add(alert.id);
      updateCount();
      announce(`${turbineName(alert.period)}: ${ACTION_LABEL[action].toLowerCase()}.`);
      done(action);
    } catch (e) {
      error.textContent = `Not saved. ${e.message}`;
      button.textContent = label;
      controls.forEach(c => { c.disabled = false; });
      box.removeAttribute('aria-busy');
    }
  };
  if (last && last.action !== 'open') {
    const undo = decidedNow.has(alert.id);
    box.append(
      el('div', { class: 'decided' },
        el('span', { class: 'tick', 'aria-hidden': 'true' }, '✓'),
        el('div', {},
          el('strong', {}, ACTION_LABEL[last.action]),
          el('span', { class: 'caption' }, ` · ${formatMoment(last.created)} UTC`),
          last.comment ? el('p', { class: 'caption' }, `“${last.comment}”`) : null),
        el('button', { type: 'button', class: 'btn plain',
          onclick: event => save(event.currentTarget, 'open') }, undo ? 'Undo' : 'Reopen')),
      error);
    return box;
  }
  const reasonSelect = el('select', {}, el('option', { value: '' }, 'Choose a reason'),
    ...DISMISS_REASONS.map(r => el('option', { value: r }, r)));
  const note = el('textarea', { rows: 2, maxLength: 2000, placeholder: 'Add a note (optional)', 'aria-label': 'Note' });
  const reason = el('div', { class: 'reason', hidden: true },
    el('label', { class: 'field' }, 'Why is this not a problem?', reasonSelect),
    note,
    el('div', { class: 'actions' },
      el('button', { type: 'button', class: 'btn', onclick: event => {
        if (!reasonSelect.value) { error.textContent = 'Choose a reason first.'; reasonSelect.focus(); return; }
        save(event.currentTarget, 'dismissed', [reasonSelect.value, note.value.trim()].filter(Boolean).join(': '));
      } }, 'Save decision'),
      el('button', { type: 'button', class: 'btn plain', onclick: () => {
        reason.hidden = true; buttons.hidden = false; error.textContent = ''; buttons.lastChild.focus();
      } }, 'Cancel')));
  const buttons = el('div', { class: 'actions' },
    el('button', { type: 'button', class: 'btn', onclick: event => save(event.currentTarget, 'inspect') }, 'Send a technician'),
    el('button', { type: 'button', class: 'btn secondary', onclick: event => save(event.currentTarget, 'acknowledged') }, 'Seen it, keep watching'),
    el('button', { type: 'button', class: 'btn plain', onclick: () => { buttons.hidden = true; reason.hidden = false; reasonSelect.focus(); } }, 'Not a problem…'));
  box.append(buttons, reason, error);
  return box;
}

// ---------- Fleet ----------
function finding(p) {
  const alert = p.alerts[0];
  if (!alert || !alert.top?.length) return 'Several small changes, nothing stands out yet.';
  return `Biggest changes: ${listText(alert.top.slice(0, 3))}.`;
}
function renderFleet() {
  const farm = $('filter-farm').value;
  const periods = fleet.periods.filter(p => !farm || p.farm === farm);
  const counts = fleetCounts(periods, reviews);
  for (const key of ['alarm', 'watch', 'normal']) $(`count-${key}`).textContent = counts[key];
  $('count-total').textContent = periods.length;
  const by = status => periods.filter(p => statusOf(p.max_crit) === status);
  const waiting = p => p.alerts.some(a => awaitingReview(reviews, a.id));
  const urgent = by('alarm').sort((a, b) => waiting(b) - waiting(a) || b.max_crit - a.max_crit);
  const shown = showAllInspect ? urgent : urgent.slice(0, SHOWN);
  $('inspect-cards').replaceChildren(...shown.map(p => el('a', { class: 'card turbine-card', href: `#/period/${p.key}` },
    el('div', { class: 'top' }, pill('alarm'), el('span', {}, p.alerts[0] ? `Since ${formatMoment(p.alerts[0].start)}` : span(p))),
    el('div', { class: 'name' }, turbineName(p)),
    el('div', { class: 'finding' }, finding(p)),
    el('div', { class: 'foot' },
      el('span', {}, el('span', { class: 'caption' }, 'Check first: '), el('strong', {}, p.look_first || 'see details')),
      el('span', { class: 'open' }, waiting(p) ? 'Decide' : 'Open')))));
  $('inspect-more').hidden = urgent.length <= SHOWN;
  $('inspect-more').textContent = showAllInspect ? 'Show fewer' : `Show all ${urgent.length}`;
  const row = p => el('a', { class: 'row', href: `#/period/${p.key}` },
    pill(statusOf(p.max_crit)), el('strong', {}, turbineName(p)),
    el('span', { class: 'text' }, statusOf(p.max_crit) === 'watch' ? `Small changes building · ${span(p)}` : span(p)));
  const watch = by('watch').sort((a, b) => b.max_crit - a.max_crit);
  const watchShown = showAllWatch ? watch : watch.slice(0, SHOWN);
  $('watch-rows').replaceChildren(...(watch.length ? watchShown.map(row) : [el('p', { class: 'row caption' }, 'Nothing to keep an eye on.')]),
    watch.length > SHOWN ? el('button', { type: 'button', class: 'more', onclick: () => { showAllWatch = !showAllWatch; renderFleet(); } },
      showAllWatch ? 'Show fewer' : `Show all ${watch.length}`) : null);
  const normal = by('normal');
  $('normal-summary').textContent = `${normal.length} periods stayed normal`;
  $('normal-rows').replaceChildren(...normal.map(row));
}

// ---------- Turbine period ----------
function peakIndex(crit) {
  let best = 0;
  for (let i = 1; i < crit.length; i++) if (crit[i] > crit[best]) best = i;
  return best;
}
async function showPeriod(key) {
  stop();
  const summary = fleet.periods.find(p => p.key === key);
  if (!summary) { message('That turbine period is not in the replay data.', true); location.hash = '#/'; return; }
  message('Loading turbine…');
  period = await periodData(key);
  period.summary = summary;
  period.times = period.hours.map(Date.parse);
  message('');
  $('period-eyebrow').textContent = `${turbineName(period)} · ${span(summary)}`;
  const keys = Object.keys(period.series);
  $('sensor').replaceChildren(...keys.map(i => el('option', { value: i }, period.sensors[i].description)));
  $('cursor').max = period.hours.length - 1;
  $('how-decided').open = false;
  const peak = peakIndex(period.crit);
  // Chart the strongest finding first, falling back to any sensor with a series.
  sensor = String((period.top[peak] || []).map(([i]) => i).find(i => period.series[i]) ?? keys[0] ?? '') || null;
  if (sensor) $('sensor').value = sensor;
  setCursor(peak);
  renderWhatNext();
}
function raisedAt(index) {
  return period.alerts.filter(a => Date.parse(a.start) <= period.times[index]).at(-1) ?? null;
}
function setCursor(index) {
  cursor = Math.max(0, Math.min(index, period.hours.length - 1));
  $('cursor').value = cursor;
  const crit = period.crit[cursor], status = statusOf(crit, period.alarm_level);
  $('clock').textContent = `Showing ${formatMoment(period.hours[cursor])} UTC`;
  $('crit-value').textContent = `${Math.min(crit, period.alarm_level)} / ${period.alarm_level}`;
  $('crit-bar').style.width = `${Math.min(100, crit / period.alarm_level * 100)}%`;
  $('period-status').replaceChildren(pill(status));
  const top = (period.top[cursor] || []).map(([index]) => period.sensors[index]);
  const component = top.find(s => s.component !== 'Operating conditions')?.component;
  const raised = raisedAt(cursor);
  $('period-title').textContent = status === 'normal'
    ? 'Behaving normally at this time'
    : status === 'watch' ? 'Small changes are building' : `${component ?? 'Several'} readings differ from normal`;
  $('period-summary').textContent = raised
    ? `Unusual since ${formatMoment(raised.start)}. Showing ${formatMoment(period.hours[cursor])}.`
    : `Showing ${formatMoment(period.hours[cursor])}.`;
  renderFindings();
  renderScoreChart();
  renderSensorChart();
  renderDecision();
}
function renderFindings() {
  const top = (period.top[cursor] || []).slice(0, 3);
  $('noticed-note').textContent = period.crit[cursor] < WATCH
    ? 'No warning at this time. These are the largest small differences.'
    : 'The sensors furthest from what is normal for this turbine.';
  $('findings').replaceChildren(...top.map(([index]) => {
    const info = period.sensors[index];
    const series = period.series[index];
    const actual = series?.actual[cursor], expected = series?.expected[cursor];
    const context = info.component === 'Operating conditions';
    const selected = String(index) === sensor;
    return el('div', { class: `card${selected ? ' selected' : ''}`, style: 'padding: 16px 20px' },
      el('button', { type: 'button', class: 'finding-row', 'aria-pressed': String(selected),
        onclick: () => { if (series) { sensor = String(index); $('sensor').value = sensor; renderFindings(); renderSensorChart(); } } },
        el('span', { class: 'what' }, el('strong', {}, info.description),
          el('span', {}, describeDeviation(actual, expected, info.unit)),
          el('small', {}, context ? 'Operating conditions: context, not equipment health' : info.component)),
        Number.isFinite(actual) && Number.isFinite(expected)
          ? el('span', { class: 'nums' },
            el('div', {}, el('span', {}, 'Now'), el('strong', { class: 'now' }, formatValue(actual, info.unit))),
            el('div', {}, el('span', {}, 'Normal'), el('strong', {}, formatValue(expected, info.unit))))
          : null));
  }));
}
function renderDecision() {
  const alert = raisedAt(cursor) ?? period.alerts[0];
  if (!alert) {
    $('period-decision').replaceChildren(el('p', { class: 'caption' }, 'No alert was raised in this period, so there is nothing to decide.'));
    return;
  }
  const full = { ...period.summary.alerts.find(a => a.id === alert.id), period: period.summary };
  $('period-decision').replaceChildren(
    el('p', { class: 'caption' }, `Alert raised ${formatMoment(alert.start)}`),
    decisionControls(full, () => { renderDecision(); focusDecision($('period-decision')); }));
}
// Keep keyboard and screen-reader users on the decision they just changed.
function focusDecision(container) {
  container.querySelector('.decision button')?.focus({ preventScroll: true });
}
function renderWhatNext() {
  const fault = period.fault;
  $('what-next').hidden = !fault;
  if (fault) {
    $('what-next-text').textContent = `This is historical data, so we know a fault was recorded: ${fault.description} (from ${formatMoment(fault.start)}). ${describeLead(warningLead(period.alerts, fault))}`;
  }
}
function frame(width, height, label) {
  return svg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': label });
}
function renderScoreChart() {
  const W = 640, H = 200, L = 12, R = 12, T = 10, B = 26;
  const t0 = period.times[0], t1 = period.times.at(-1);
  const high = upperBound(period.score, period.threshold);
  const x = t => L + (t - t0) / (t1 - t0 || 1) * (W - L - R);
  const y = v => T + (1 - Math.min(v, high) / high) * (H - T - B);
  const chart = frame(W, H, 'Unusualness over time with the threshold and recorded fault period');
  if (period.fault) {
    const a = x(Math.max(t0, Date.parse(period.fault.start))), b = x(Math.min(t1, Date.parse(period.fault.end)));
    if (b > a) chart.append(svg('rect', { x: a, y: T, width: b - a, height: H - T - B, fill: '#E3F2FB' }));
  }
  chart.append(svg('line', { x1: L, x2: W - R, y1: y(period.threshold), y2: y(period.threshold), stroke: '#01507F', 'stroke-dasharray': '5 4' }));
  const shown = period.score.map((v, i) => (i <= cursor ? v : null));
  chart.append(svg('path', { d: linePath(period.times, shown, x, y), fill: 'none', stroke: '#0272B8', 'stroke-width': 2 }));
  chart.append(svg('line', { x1: x(period.times[cursor]), x2: x(period.times[cursor]), y1: T, y2: H - B, stroke: '#0F2133', 'stroke-dasharray': '2 3' }));
  chart.append(svg('text', { x: L, y: H - 8 }, formatDate(period.hours[0])));
  chart.append(svg('text', { x: W - R, y: H - 8, 'text-anchor': 'end' }, formatDate(period.hours.at(-1))));
  $('score-chart').replaceChildren(chart);
}
function renderSensorChart() {
  if (sensor === null || !period.series[sensor]) { $('sensor-chart').replaceChildren(el('p', { class: 'caption' }, 'Choose a sensor.')); return; }
  const info = period.sensors[sensor];
  $('sensor-title').textContent = info.description;
  const { actual, expected } = period.series[sensor];
  const W = 720, H = 240, L = 64, R = 12, T = 14, B = 28;
  const values = [...actual, ...expected].filter(Number.isFinite);
  let low = Math.min(...values), high = Math.max(...values);
  const pad = (high - low) * 0.1 || 1; low -= pad; high += pad;
  const t0 = period.times[0], t1 = period.times.at(-1);
  const x = t => L + (t - t0) / (t1 - t0 || 1) * (W - L - R);
  const y = v => T + (high - v) / (high - low) * (H - T - B);
  const chart = frame(W, H, `${info.description}: actual reading and normal level`);
  for (const alert of period.alerts) {
    const t = Date.parse(alert.start);
    chart.append(svg('line', { x1: x(t), x2: x(t), y1: T, y2: H - B, stroke: '#0F2133', 'stroke-width': 2 }));
  }
  chart.append(svg('line', { x1: x(period.times[cursor]), x2: x(period.times[cursor]), y1: T, y2: H - B, stroke: '#50647A', 'stroke-dasharray': '2 3' }));
  chart.append(svg('path', { d: linePath(period.times, expected, x, y), fill: 'none', stroke: '#6B7F93', 'stroke-dasharray': '6 4', 'stroke-width': 2 }));
  chart.append(svg('path', { d: linePath(period.times, actual, x, y), fill: 'none', stroke: '#0272B8', 'stroke-width': 2.5 }));
  const unit = cleanUnit(info.unit);
  chart.append(svg('text', { x: 4, y: T + 10 }, `${high.toPrecision(3)} ${unit}`));
  chart.append(svg('text', { x: 4, y: H - B }, `${low.toPrecision(3)} ${unit}`));
  chart.append(svg('text', { x: L, y: H - 8 }, formatDate(period.hours[0])));
  chart.append(svg('text', { x: W - R, y: H - 8, 'text-anchor': 'end' }, formatDate(period.hours.at(-1))));
  $('sensor-chart').replaceChildren(chart);
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
  $('show-awaiting').setAttribute('aria-pressed', String(!showAllAlerts));
  $('show-all').setAttribute('aria-pressed', String(showAllAlerts));
  const alerts = allAlerts().filter(a => showAllAlerts || awaitingReview(reviews, a.id) || a.id === focus)
    .sort((a, b) => b.peak - a.peak);
  $('alerts-empty').hidden = alerts.length > 0;
  $('alert-list').replaceChildren(...alerts.map(alert => alertCard(alert)));
  if (focus) document.getElementById(`alert-${focus}`)?.scrollIntoView({ block: 'center' });
}
// A decision redraws only its own card, so the list never jumps; decided cards leave
// the To decide list the next time it is drawn.
function alertCard(alert) {
  const history = reviews.filter(r => r.alert === alert.id);
  const last = latestReview(reviews, alert.id);
  const decided = Boolean(last && last.action !== 'open');
  const card = el('article', { class: `card alert-card${decided ? ' is-decided' : ''}`, id: `alert-${alert.id}` },
    el('div', { class: 'head' },
      decided ? pill('done', 'Decided') : pill('alarm'),
      el('div', { class: 'what' },
        el('a', { href: `#/period/${alert.period.key}` }, turbineName(alert.period)),
        el('span', {}, `Check first: ${alert.component}. ${alert.top?.length ? `Biggest changes: ${listText(alert.top)}.` : ''}`)),
      el('span', { class: 'caption' }, `Since ${formatMoment(alert.start)}`)),
    decisionControls(alert, () => {
      const fresh = alertCard(alert);
      card.replaceWith(fresh);
      focusDecision(fresh);
    }),
    history.length > 1 ? el('div', { class: 'history' }, el('strong', {}, 'History'), ...history.map(r =>
      el('span', {}, `${formatMoment(r.created)} · ${ACTION_LABEL[r.action]}${r.comment ? ` · “${r.comment}”` : ''}`))) : null);
  return card;
}

// ---------- Results ----------
function renderResults() {
  const cases = fleet.periods.filter(p => p.recorded_fault).map(p => ({ p, lead: warningLead(p.alerts, p.recorded_fault) }))
    .sort((a, b) => (b.lead ?? -1e9) - (a.lead ?? -1e9));
  $('case-rows').replaceChildren(...cases.map(({ p, lead }) => el('tr', {},
    el('td', {}, el('a', { href: `#/period/${p.key}` }, turbineName(p))),
    el('td', { class: 'text', title: p.recorded_fault.description }, p.recorded_fault.description.length > 56
      ? `${p.recorded_fault.description.slice(0, 55)}…` : p.recorded_fault.description),
    el('td', { class: 'text' }, p.alerts[0] ? formatMoment(p.alerts[0].start) : 'None'),
    el('td', { class: 'text' }, lead === null ? 'No alarm' : describeLead(lead).replace('Flagged ', '').replace(' the recorded fault started.', '')))));
}

// ---------- Routing ----------
async function route() {
  const [, view = '', id = ''] = location.hash.split('/');
  for (const name of ['fleet', 'period', 'alerts', 'results']) $(`view-${name}`).hidden = true;
  for (const link of document.querySelectorAll('[data-nav]')) link.removeAttribute('aria-current');
  const current = name => document.querySelector(`[data-nav=${name}]`)?.setAttribute('aria-current', 'page');
  try {
    if (view === 'period' && id) { $('view-period').hidden = false; await showPeriod(decodeURIComponent(id)); }
    else if (view === 'alerts') {
      stop(); $('view-alerts').hidden = false; current('alerts');
      renderAlerts(id ? decodeURIComponent(id) : null);
    } else if (view === 'results') {
      stop(); $('view-results').hidden = false; current('results'); renderResults();
    } else {
      stop(); $('view-fleet').hidden = false; current('fleet'); renderFleet();
    }
    if (!id) window.scrollTo(0, 0);
  } catch (error) { message(error.message, true); }
}

$('filter-farm').onchange = renderFleet;
$('inspect-more').onclick = () => { showAllInspect = !showAllInspect; renderFleet(); };
$('show-awaiting').onclick = () => { showAllAlerts = false; renderAlerts(); };
$('show-all').onclick = () => { showAllAlerts = true; renderAlerts(); };
$('play').onclick = togglePlay;
$('cursor').oninput = () => { stop(); setCursor(Number($('cursor').value)); };
$('sensor').onchange = () => { sensor = $('sensor').value; renderFindings(); renderSensorChart(); };
$('logout').onclick = async () => {
  await fetch('/api/auth/logout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  location.replace('/');
};
window.addEventListener('hashchange', route);

try {
  message('Loading replay data…');
  const [{ user }, data, saved] = await Promise.all([api('/api/auth/me'), api('care-data/fleet.json'), api('/api/care/reviews')]);
  fleet = data; reviews = saved.reviews;
  $('role').textContent = ROLE_BADGE[user.role];
  showForRole(document.querySelectorAll('[data-role]'), user.role);
  message('');
  updateCount();
  await route();
} catch (error) {
  message(`${error.message} Generate the replay data with scripts/build_care_app_data.py, then reload.`, true);
}
