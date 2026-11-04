import test from 'node:test';
import assert from 'node:assert/strict';
import {
  awaitingReview, cleanUnit, describeDeviation, describeLead, fleetCounts, formatDate, formatHours,
  formatValue, indexAt, latestReview, linePath, STATUS_LABEL, statusOf, upperBound, warningLead,
} from '../care-model.js';

test('status follows the warning counter', () => {
  assert.equal(statusOf(0), 'normal');
  assert.equal(statusOf(23), 'normal');
  assert.equal(statusOf(24), 'watch');
  assert.equal(statusOf(71), 'watch');
  assert.equal(statusOf(72), 'alarm');
});

test('latest decision wins and reopening puts an alert back in the queue', () => {
  const reviews = [
    { alert: 'C-47-1', action: 'inspect' },
    { alert: 'A-0-1', action: 'dismissed' },
    { alert: 'C-47-1', action: 'open' },
  ];
  assert.equal(latestReview(reviews, 'C-47-1').action, 'open');
  assert.equal(awaitingReview(reviews, 'C-47-1'), true);
  assert.equal(awaitingReview(reviews, 'A-0-1'), false);
  assert.equal(awaitingReview(reviews, 'B-7-1'), true);
});

test('fleet counts use peak status and unreviewed alerts', () => {
  const periods = [
    { max_crit: 500, alerts: [{ id: 'a' }, { id: 'b' }] },
    { max_crit: 30, alerts: [] },
    { max_crit: 2, alerts: [] },
  ];
  assert.deepEqual(fleetCounts(periods, [{ alert: 'a', action: 'acknowledged' }]),
    { alarm: 1, watch: 1, normal: 1, awaiting: 1 });
});

test('indexAt finds the last time at or before the cursor', () => {
  const times = [10, 20, 30];
  assert.equal(indexAt(times, 5), -1);
  assert.equal(indexAt(times, 20), 1);
  assert.equal(indexAt(times, 29), 1);
  assert.equal(indexAt(times, 99), 2);
});

test('line paths break at missing values and long time gaps', () => {
  const id = v => v;
  const hour = 3600e3;
  const times = [0, hour, 2 * hour, 10 * hour, 11 * hour];
  assert.equal(linePath(times, [1, 2, null, 4, 5], id, id),
    `M0.0,1.0 L${hour}.0,2.0 M${10 * hour}.0,4.0 L${11 * hour}.0,5.0`);
});

test('chart bound keeps the threshold visible and ignores rare spikes', () => {
  const values = Array.from({ length: 100 }, (_, i) => (i === 99 ? 1000 : 1));
  assert.ok(upperBound(values, 2) < 10);
  assert.ok(upperBound([], 2) > 2);
});

test('durations read naturally', () => {
  assert.equal(formatHours(5), '5 h');
  assert.equal(formatHours(103), '4 days');
});

test('statuses use plain language', () => {
  assert.equal(STATUS_LABEL[statusOf(80)], 'Inspect now');
  assert.equal(STATUS_LABEL[statusOf(30)], 'Keep an eye on');
});

test('units and values are shown cleanly', () => {
  assert.equal(cleanUnit('�C'), '°C');
  assert.equal(cleanUnit('Celsius'), '°C');
  assert.equal(cleanUnit('ï¿½C'), '°C');
  assert.equal(cleanUnit('kWh'), 'kWh');
  assert.equal(cleanUnit('A'), 'A');
  assert.equal(formatValue(87.04, 'Celsius'), '87 °C');
  assert.equal(formatValue(7.797, 'A'), '7.8 A');
});

test('deviations are described against normal', () => {
  assert.equal(describeDeviation(87, 63.76, '�C'), 'About 23 °C higher than normal.');
  assert.equal(describeDeviation(89.17, 108, 'Celsius'), 'About 19 °C lower than normal.');
  assert.equal(describeDeviation(12, 10, 'A'), 'About 20% higher than normal.');
  assert.equal(describeDeviation(10.1, 10, 'A'), 'Close to normal right now.');
  assert.equal(describeDeviation(null, 10, 'A'), 'Differs from its normal pattern.');
});

test('warning timing is honest about alarms after the fault started', () => {
  const fault = { start: '2023-08-06T06:10:00' };
  assert.equal(warningLead([], fault), null);
  const early = warningLead([{ start: '2023-08-04T18:10:00' }], fault);
  assert.equal(early, 36);
  assert.equal(describeLead(early), 'Flagged 36 h before the recorded fault started.');
  assert.equal(describeLead(-45), 'Flagged 45 h after the recorded fault started.');
  assert.equal(describeLead(null), 'No alarm was raised in this period.');
  assert.equal(formatDate('2023-08-04T18:00:00'), '4 Aug 2023');
  // Saved alert decisions carry an explicit UTC offset from the server.
  assert.equal(formatDate('2026-11-04T09:12:00.123456+00:00'), '4 Nov 2026');
});
