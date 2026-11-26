// Pure helpers for the CARE early warning pages; no DOM access, unit-tested.

export const ALARM = 72;
// Display rule only: a third of the alarm level is shown as "Watch".
export const WATCH = 24;

export function statusOf(crit, alarm = ALARM) {
  if (crit >= alarm) return 'alarm';
  if (crit >= WATCH) return 'watch';
  return 'normal';
}

export const STATUS_LABEL = { alarm: 'Inspect now', watch: 'Keep an eye on', normal: 'Normal' };

export const ACTION_LABEL = {
  inspect: 'Technician sent',
  acknowledged: 'Seen, keep watching',
  dismissed: 'Not a problem',
  open: 'Reopened',
};

export const DISMISS_REASONS = [
  'Planned maintenance or test was running',
  'Known sensor problem',
  'Unusual weather or grid conditions',
  'Checked on site, nothing wrong',
  'Other',
];

// The replay data carries some units with a broken degree sign; show a clean one.
export function cleanUnit(unit = '') {
  const text = String(unit).trim();
  // e.g. "Celsius", "°C", or a mis-decoded degree sign such as "ï¿½C".
  if (/^(celsius|deg ?c|[^a-z0-9 ]{0,4}c)$/i.test(text)) return '°C';
  return text;
}

export function formatValue(value, unit) {
  const digits = Math.abs(value) >= 10 ? 0 : 1;
  const clean = cleanUnit(unit);
  return `${value.toFixed(digits)}${clean ? (clean === '°C' ? ' °C' : ` ${clean}`) : ''}`;
}

// No turbine part runs outside this range. A temperature beyond it is almost always a failed
// or disconnected sensor (a Pt100 sensor reads its 850 °C full scale), not a real reading.
export const TEMPERATURE_RANGE = [-60, 300];

export function likelySensorFault(value, unit) {
  if (!Number.isFinite(value) || cleanUnit(unit) !== '°C') return false;
  return value < TEMPERATURE_RANGE[0] || value > TEMPERATURE_RANGE[1];
}

// ---------- Technician teams ----------
// A suggestion for who to send, from the sensors that moved most; the engineer decides.
export const TEAM_LABEL = {
  mechanical: 'Mechanical',
  electrical: 'Electrical',
  hydraulics: 'Hydraulics and cooling',
  pitch: 'Pitch and hub',
  instrumentation: 'Instrumentation',
};

// First match wins, so specific parts come before broad words such as "current".
// 'context' marks operating conditions (weather, wind), which need no technician.
const TEAM_RULES = [
  ['electrical', /transformer|transformator/],
  ['hydraulics', /hydraulic/],
  ['mechanical', /gearbox|gear oil|rotor bearing|main bearing/],
  ['context', /\bwind\b|wind ?speed|ambient|outside temperature|anemometer/],
  ['electrical', /\bpower\b|reactive/],
  ['mechanical', /generator.*bearing|bearing.*generator/],
  ['electrical', /generator|slip ring|stator/],
  ['mechanical', /tower|vibration/],
  ['pitch', /pitch|blade|\baxis\b|batter|\bhub\b|spinner/],
  ['electrical', /converter|inverter|igbt|\bvcs\b|\bvcp\b|choke/],
  ['hydraulics', /cooling|cooler|water|aerat/],
  ['electrical', /voltage|current|grid|frequency|consumption|busbar|\bphase\b/],
];

// Team for a sensor by its description; null when no rule fits.
export function sensorTeam(description = '') {
  const text = String(description).toLowerCase();
  return TEAM_RULES.find(([, pattern]) => pattern.test(text))?.[0] ?? null;
}

// Team for one reading: an impossible value goes to instrumentation, not the part.
export function readingTeam(description, value, unit) {
  return likelySensorFault(value, unit) ? 'instrumentation' : sensorTeam(description);
}

// Most common team among ranked findings; ties go to the higher-ranked sensor.
export function suggestTeam(teams) {
  const counted = new Map();
  for (const team of teams) {
    if (team && team !== 'context') counted.set(team, (counted.get(team) ?? 0) + 1);
  }
  let best = null;
  for (const [team, count] of counted) if (!best || count > counted.get(best)) best = team;
  return best;
}

export function alertTeam(alert) {
  return suggestTeam((alert.top ?? []).map(sensorTeam));
}

// Plain-language comparison of a reading with the model's expected value.
export function describeDeviation(actual, expected, unit) {
  if (likelySensorFault(actual, unit)) return 'Out of range for this sensor, so not a real temperature.';
  if (!Number.isFinite(actual) || !Number.isFinite(expected)) return 'Differs from its normal pattern.';
  const gap = actual - expected;
  const scale = Math.max(Math.abs(expected), 1);
  if (Math.abs(gap) / scale < 0.02) return 'Close to normal right now.';
  const direction = gap > 0 ? 'higher' : 'lower';
  if (cleanUnit(unit) === '°C') return `About ${Math.round(Math.abs(gap))} °C ${direction} than normal.`;
  return `About ${Math.round(Math.abs(gap) / scale * 100)}% ${direction} than normal.`;
}

// Signed hours from the first alarm to the recorded fault start (positive = warned before).
export function warningLead(alerts, fault) {
  if (!fault || !alerts.length) return null;
  return (Date.parse(fault.start) - Date.parse(alerts[0].start)) / 3600e3;
}

export function describeLead(hours) {
  if (hours === null) return 'No alarm was raised in this period.';
  if (hours >= 0) return `Flagged ${formatHours(hours)} before the recorded fault started.`;
  return `Flagged ${formatHours(-hours)} after the recorded fault started.`;
}

export function formatDate(iso) {
  // Replay times have no zone; saved decisions carry one (e.g. +00:00). Both are UTC.
  const date = new Date(/(Z|[+-]\d\d:\d\d)$/.test(iso) ? iso : `${iso}Z`);
  return date.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
}

export function formatMoment(iso) {
  return `${formatDate(iso)}, ${iso.slice(11, 16)}`;
}

export function latestReview(reviews, alertId) {
  for (let i = reviews.length - 1; i >= 0; i--) if (reviews[i].alert === alertId) return reviews[i];
  return null;
}

export function awaitingReview(reviews, alertId) {
  const last = latestReview(reviews, alertId);
  return !last || last.action === 'open';
}

export function fleetCounts(periods, reviews) {
  const counts = { alarm: 0, watch: 0, normal: 0, awaiting: 0 };
  for (const period of periods) {
    counts[statusOf(period.max_crit)]++;
    for (const alert of period.alerts) if (awaitingReview(reviews, alert.id)) counts.awaiting++;
  }
  return counts;
}

// Last index whose time is at or before `time`; -1 when none.
export function indexAt(times, time) {
  let low = 0, high = times.length - 1, found = -1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (times[mid] <= time) { found = mid; low = mid + 1; } else high = mid - 1;
  }
  return found;
}

// SVG path through finite values; nulls and time gaps longer than `gap` start new segments.
export function linePath(times, values, x, y, gap = 3 * 3600e3) {
  let path = '', open = false;
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (value === null || !Number.isFinite(value)) { open = false; continue; }
    const jump = i > 0 && times[i] - times[i - 1] > gap;
    path += `${open && !jump ? 'L' : 'M'}${x(times[i]).toFixed(1)},${y(value).toFixed(1)} `;
    open = true;
  }
  return path.trim();
}

export function upperBound(values, threshold) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  const high = finite.length ? finite[Math.floor((finite.length - 1) * 0.98)] : threshold;
  return Math.max(high, threshold * 1.6) * 1.08;
}

export function formatTime(iso) {
  return iso.replace('T', ' ').slice(0, 16);
}

export function formatHours(hours) {
  if (hours < 48) return `${Math.round(hours)} h`;
  return `${Math.round(hours / 24)} days`;
}
