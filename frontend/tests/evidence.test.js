import test from 'node:test';
import assert from 'node:assert/strict';
import { describeEvent, scoreSegments, validateEvidence } from '../evidence-model.js';

test('score curves break at unavailable history', () => {
  const rows = ['2019-02-01T00:00Z', '2019-02-01T00:05Z', '2019-02-01T01:00Z'].map(t => ({ t }));
  assert.deepEqual(scoreSegments(rows).map(s => s.length), [2, 1]);
  assert.deepEqual(scoreSegments([]), []);
});

test('insufficient evidence never receives a missed-warning explanation', () => {
  const result = describeEvent({ status: 'insufficient', scored_points: 0 });
  assert.match(result.body, /not counted as a detection or a miss/);
  const partial = describeEvent({ status: 'insufficient', scored_points: 5, alert_points: 1 });
  assert.match(partial.body, /crossings are visible/);
  assert.match(describeEvent({ status: 'warned', alert_points: 1 }).body, /single crossing is weaker/);
});

test('rejects fabricated or contradictory score decisions', () => {
  const data = { schema_version: 1, model: 'test', threshold: 1, summary: {}, provenance: {},
    scores: [{ t: '2019-02-01T00:00Z', ratio: .5, alert: true }],
    events: [{ id: 1, fault: '2019-02-01T01:00Z', status: 'missed', coverage: 1 }] };
  assert.throws(() => validateEvidence(data), /Invalid saved score/);
  data.scores[0].alert = false;
  assert.equal(validateEvidence(data), data);
});
