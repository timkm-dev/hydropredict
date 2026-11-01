import test from 'node:test';
import assert from 'node:assert/strict';

// Exercise page initialization and controls without a browser or real dataset.
function fixture() {
  const events = ['warned', 'missed', 'insufficient'].map((status, i) => ({
    id: i + 1, fault: `2019-02-0${i + 2}T00:00:00Z`, group: i + 1, status,
    eligible: status !== 'insufficient', coverage: status === 'insufficient' ? 0 : 1,
    scored_points: status === 'insufficient' ? 0 : 72,
    latest_gap_minutes: status === 'insufficient' ? null : 5,
    alert_points: status === 'warned' ? 1 : 0,
    first_alert: status === 'warned' ? '2019-02-01T23:55:00Z' : null,
    lead_minutes: status === 'warned' ? 5 : null,
    peak_ratio: status === 'insufficient' ? null : status === 'warned' ? 1.1 : .6,
  }));
  return { schema_version: 1, model: 'synthetic-test', threshold: 1,
    period: ['2019-02-01T00:00Z', '2019-05-01T00:00Z'], provenance: {}, events,
    scores: [{ t: '2019-02-01T23:55:00Z', ratio: 1.1, alert: true }],
    summary: { faults: 3, eligible: 2, warned: 1, missed: 1, insufficient: 1,
      score_points: 1, observations: 1, alert_points: 1, episodes: 1,
      associated_alert_points: 1, outside_alert_points: 0, nominal_points: 1,
      nominal_alert_points: 0 } };
}

async function withPage(response, action, suffix) {
  const saved = Object.fromEntries(['document', 'window', 'location', 'history', 'fetch'].map(k => [k, globalThis[k]]));
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, { innerHTML: '', textContent: '', hidden: false, value: '6' });
    return elements.get(id);
  };
  globalThis.document = { getElementById: get, querySelectorAll: () => [], addEventListener: () => {} };
  globalThis.window = { addEventListener: () => {} };
  globalThis.location = { hash: '' };
  globalThis.history = { replaceState: (_state, _title, hash) => { location.hash = hash; } };
  globalThis.fetch = async () => response;
  try {
    await import(`../evidence.js?test=${suffix}`);
    await new Promise(resolve => setImmediate(resolve));
    await action(get);
  } finally {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete globalThis[key];
      else globalThis[key] = value;
    }
  }
}

test('page renders outcomes, navigation and missing-score chart', async () => {
  await withPage({ ok: true, json: async () => fixture() }, async get => {
    assert.equal(get('load-status').hidden, true);
    assert.match(get('event-title').textContent, /Warning observed/);
    assert.match(get('event-chart').innerHTML, /Anomaly scores around fault 1/);
    assert.equal(get('previous').disabled, true);
    get('next').onclick();
    assert.match(get('event-title').textContent, /Missed warning/);
    get('history').value = '12';
    get('history').onchange();
    assert.match(get('event-chart').innerHTML, /-12h/);
    get('next').onclick();
    assert.match(get('event-explanation').innerHTML, /cannot assess/);
    assert.match(get('event-chart').innerHTML, /No eligible model scores/);
    assert.equal(get('next').disabled, true);
  }, 'loaded');
});

test('missing local evidence gives setup instructions, never demo results', async () => {
  await withPage({ ok: false }, async get => {
    assert.match(get('load-status').innerHTML, /map_anomalies_to_faults.py/);
    assert.match(get('load-status').innerHTML, /No example results are substituted/);
    assert.equal(get('stats').innerHTML, '');
  }, 'missing');
});
