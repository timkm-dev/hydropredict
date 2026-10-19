import { CHANNELS } from './data.js';

export async function getModel() {
  const response = await fetch('/api/model', { cache: 'no-store' });
  if (!response.ok) throw new Error('Start the LSTM dashboard server to enable inference.');
  return response.json();
}

export async function scoreTelemetry(records) {
  const response = await fetch('/api/score', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ records }),
  });
  let result;
  try { result = await response.json(); }
  catch { throw new Error('Inference server is unavailable. Start the Python dashboard server.'); }
  if (!response.ok) throw new Error(result.error || 'The model could not assess these readings.');
  return result;
}

// Collapse adjacent positive decisions into review episodes without changing the
// frozen point-alert rule. A normal score or cadence gap always ends an episode.
export function scoreEvents(result, unitId) {
  const events = [];
  let episode = null;
  let previous = null;
  for (const score of result.scores) {
    const delta = previous ? (Date.parse(score.t) - Date.parse(previous.t)) / 1000 : Infinity;
    if (!score.is_alert) { episode = null; previous = score; continue; }
    const continuous = episode && delta >= result.model.minimum_step_seconds && delta <= result.model.maximum_step_seconds && previous.segment_id === score.segment_id;
    if (!continuous) {
      episode = { id: `LSTM-${events.length + 1}`, unitId, time: score.t, endTime: score.t, peakTime: score.t, peak: score.anomaly_score, count: 0, severity: 'Inspect', acknowledged: false, errors: {} };
      events.push(episode);
    }
    episode.count++;
    episode.endTime = score.t;
    if (score.anomaly_score >= episode.peak) {
      episode.peak = score.anomaly_score;
      episode.peakTime = score.t;
      episode.errors = Object.fromEntries(CHANNELS.map(channel => [channel, score[`${channel}_reconstruction_error`]]));
      episode.channel = CHANNELS.reduce((best, channel) => episode.errors[channel] > episode.errors[best] ? channel : best, 'V1');
    }
    episode.title = `LSTM flagged ${episode.count} reading${episode.count === 1 ? '' : 's'}`;
    episode.explanation = `Peak anomaly score ${episode.peak.toFixed(4)} exceeds the saved threshold ${result.model.threshold.toFixed(4)}. ${episode.channel} has the largest reconstruction error at the peak. This indicates unusual measurements, not a diagnosis or a confirmed faulty component.`;
    previous = score;
  }
  return events;
}

export function scoresInWindow(result, rows) {
  if (!result || !rows.length) return [];
  const start = Date.parse(rows[0].t), end = Date.parse(rows.at(-1).t);
  return result.scores.filter(score => Date.parse(score.t) >= start && Date.parse(score.t) <= end);
}
