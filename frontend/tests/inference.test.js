import test from 'node:test';
import assert from 'node:assert/strict';
import { scoreEvents, scoresInWindow, scoreTelemetry } from '../inference.js';

const make = (minute, alert, score = 0.5, segment = 0) => ({
  t: new Date(Date.UTC(2026, 0, 1, 0, minute)).toISOString(),
  is_alert: alert, anomaly_score: score, segment_id: segment,
  ...Object.fromEntries(Array.from({ length: 6 }, (_, i) => [`V${i + 1}_reconstruction_error`, i === 3 ? 1 : 0.1])),
});
const model = { minimum_step_seconds: 150, maximum_step_seconds: 450, threshold: 0.3 };

test('groups consecutive alerts but splits at normal decisions and missing context', () => {
  const events = scoreEvents({ model, scores: [make(0, true), make(5, true, 0.8), make(10, false), make(15, true), make(40, true), make(45, true, 0.6, 1)] }, 'unassigned');
  assert.equal(events.length, 4);
  assert.equal(events[0].count, 2);
  assert.equal(events[0].peak, 0.8);
  assert.equal(events[0].channel, 'V4');
  assert.equal(events[0].unitId, 'unassigned');
  assert.equal(events[0].acknowledged, false);
});
test('empty or below-threshold results produce no flagged episodes', () => {
  assert.deepEqual(scoreEvents({ model, scores: [] }, 'x'), []);
  assert.deepEqual(scoreEvents({ model, scores: [make(0, false)] }, 'x'), []);
});
test('filters already-scored observations without restarting sequence context', () => {
  const scores = [make(0, false), make(5, true), make(10, false)];
  assert.deepEqual(scoresInWindow({ scores }, [{ t: scores[1].t }, { t: scores[2].t }]), scores.slice(1));
  assert.deepEqual(scoresInWindow(null, []), []);
});

test('sends telemetry as JSON and surfaces server validation failures', async context => {
  context.mock.method(globalThis, 'fetch', async (path, options) => {
    assert.equal(path, '/api/score');
    assert.equal(options.method, 'POST');
    assert.equal(options.headers['Content-Type'], 'application/json');
    assert.deepEqual(JSON.parse(options.body), { records: [] });
    return { ok: false, json: async () => ({ error: 'No eligible input records.' }) };
  });
  await assert.rejects(scoreTelemetry([]), /No eligible input records/);
});
