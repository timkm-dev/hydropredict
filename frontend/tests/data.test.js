import test from 'node:test';
import assert from 'node:assert/strict';
import { demoTelemetry, parseTelemetry, windowRows, toCSV } from '../data.js';

const header = 't,V1,V2,V3,V4,V5,V6\n';
test('demo exports round-trip without losing measurements', () => {
  const rows = demoTelemetry();
  assert.equal(rows.length, 288);
  assert.deepEqual(parseTelemetry(toCSV(rows)), rows);
  assert.equal(windowRows(rows, '6').length, 72);
  assert.equal(windowRows(rows, 'all').length, 288);
});
test('imports reordered columns, quoted fields, BOM, extra metadata and naive UTC times', () => {
  const rows = parseTelemetry('\uFEFFV6,t,V1,V2,V3,V4,V5,note\r\n6,"2019-01-02 00:00:00",1,2,3,4,5,"a,b"\r\n6,2019-01-01,1,2,3,4,5,"two\nlines"');
  assert.equal(rows[0].t, '2019-01-01T00:00:00.000Z');
  assert.equal(rows[1].V6, 6);
});
test('rejects incomplete schema, missing values, invalid dates, duplicates, and non-finite values', () => {
  for (const text of [
    't,V1\n2019-01-01,1',
    header + '2019-01-01,1,,3,4,5,6',
    header + '2019-02-30,1,2,3,4,5,6',
    header + '2019-01-01T24:00:00Z,1,2,3,4,5,6',
    header + '2019-01-01,NaN,2,3,4,5,6',
    header + '2019-01-01,1e999,2,3,4,5,6',
    header + '2019-01-01,1,2,3,4,5,6\n2019-01-01T00:00:00Z,1,2,3,4,5,6',
    header + '2019-01-01,1,2,3,4,5',
    header + '"2019-01-01,1,2,3,4,5,6',
  ]) assert.throws(() => parseTelemetry(text));
});
