import test from 'node:test';
import assert from 'node:assert/strict';
import { DEMO_UNITS, unitTelemetry, demoAlerts, unitStatus, eventsInWindow } from '../units.js';
import { windowRows } from '../data.js';

test('events identify equipment independently of signal channel IDs', () => {
  const events = demoAlerts();
  assert.equal(unitStatus('unit-01', events, true).kind, 'inspect');
  assert.equal(unitStatus('unit-02', events, true).kind, 'watch');
  assert.equal(unitStatus('unit-03', events, true).kind, 'clear');
  assert.deepEqual(events.map(event => event.unitId), ['unit-01', 'unit-02']);
  assert.equal(DEMO_UNITS.every(unit => unit.channels.length === 6), true);
  assert.notDeepEqual(unitTelemetry('unit-01'), unitTelemetry('unit-02'));
});
test('reviewing an event never clears an equipment anomaly', () => {
  const events = demoAlerts();
  events.forEach(event => { event.acknowledged = true; });
  assert.equal(unitStatus('unit-01', events, true).kind, 'inspect');
  assert.equal(unitStatus('unit-02', events, true).kind, 'watch');
});
test('unassessed imports stay unknown even with no events', () => {
  assert.equal(unitStatus('unassigned', [], false).kind, 'unknown');
  assert.equal(unitStatus('unit-03', demoAlerts(), false).kind, 'unknown');
});
test('review period filters events without claiming old anomalies were resolved', () => {
  const rows = unitTelemetry('unit-01');
  assert.equal(eventsInWindow(demoAlerts(), rows).length, 2);
  const recent = eventsInWindow(demoAlerts(), windowRows(rows, '12'));
  assert.equal(recent.length, 1);
  assert.equal(recent[0].unitId, 'unit-01');
  assert.deepEqual(eventsInWindow(demoAlerts(), []), []);
});
