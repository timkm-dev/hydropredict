import { demoTelemetry } from './data.js';

// Equipment identities and assignments here are invented demo fixtures, not
// inferred from V1-V6. Each unit has its own six-channel telemetry series.
export const DEMO_UNITS = [
  { id: 'unit-01', name: 'Unit 01', location: 'Demo generating unit', channels: ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'] },
  { id: 'unit-02', name: 'Unit 02', location: 'Demo generating unit', channels: ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'] },
  { id: 'unit-03', name: 'Unit 03', location: 'Demo generating unit', channels: ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'] },
];

export function unitTelemetry(unitId) {
  return demoTelemetry().map((row, index) => {
    const copy = { ...row };
    // Remove the original fixture excursion, then apply per-unit examples.
    const shape = index > 178 && index < 198 ? Math.sin((index - 178) / 20 * Math.PI) : 0;
    DEMO_UNITS[0].channels.forEach((channel, c) => {
      copy[channel] -= shape * (c === 0 ? 8 : 1.8);
      if (unitId === 'unit-02') copy[channel] += 3;
      if (unitId === 'unit-03') copy[channel] -= 2;
    });
    if (unitId === 'unit-01') copy.V1 += shape * 8;
    if (unitId === 'unit-02' && index >= 128 && index <= 140) copy.V4 += Math.sin((index - 128) / 12 * Math.PI) * 5;
    return copy;
  });
}

export function demoAlerts() {
  return [
    { id: 'DEMO-003', unitId: 'unit-01', channel: 'V1', title: 'Sudden rise in signal V1', time: '2026-10-18T15:30:00Z', severity: 'Inspect', acknowledged: false, explanation: 'The demo signal rises sharply above its earlier pattern. This is an unusual reading, not a confirmed equipment fault.' },
    { id: 'DEMO-002', unitId: 'unit-02', channel: 'V4', title: 'Short change in signal V4', time: '2026-10-18T11:10:00Z', severity: 'Watch', acknowledged: false, explanation: 'The demo signal briefly moves away from its earlier pattern. Review the surrounding readings before deciding on an inspection.' },
  ];
}

export function unitStatus(unitId, events, evaluated) {
  if (!evaluated) return { kind: 'unknown', label: 'Not assessed', summary: 'No model result is available for this equipment.' };
  const unitEvents = events.filter(event => event.unitId === unitId);
  if (unitEvents.some(event => event.severity === 'Inspect')) return { kind: 'inspect', label: 'Anomaly flagged', summary: 'Review the unusual signal and decide whether an inspection is needed.' };
  if (unitEvents.length) return { kind: 'watch', label: 'Watch', summary: 'A signal change was flagged. Review its readings.' };
  return { kind: 'clear', label: 'No anomalies flagged', summary: 'No demo events in this time window. This does not confirm equipment health.' };
}

export function eventsInWindow(events, rows) {
  if (!rows.length) return [];
  const start = Date.parse(rows[0].t), end = Date.parse(rows.at(-1).t);
  return events.filter(event => Date.parse(event.time) >= start && Date.parse(event.time) <= end);
}
