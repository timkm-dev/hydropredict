// Pure helpers for the CARE early warning pages; no DOM access, unit-tested.

export const ALARM = 72;
// Display rule only: a third of the alarm level is shown as "Watch".
export const WATCH = 24;

export function statusOf(crit, alarm = ALARM) {
  if (crit >= alarm) return 'alarm';
  if (crit >= WATCH) return 'watch';
  return 'normal';
}

export const STATUS_LABEL = { alarm: 'Alarm', watch: 'Watch', normal: 'Normal' };

export const ACTION_LABEL = {
  inspect: 'Sent for inspection',
  acknowledged: 'Acknowledged',
  dismissed: 'Dismissed',
  open: 'Reopened',
};

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
