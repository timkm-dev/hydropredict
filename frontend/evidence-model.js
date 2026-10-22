export const LABELS = { warned: 'Warning observed', missed: 'Missed warning', insufficient: 'Insufficient data' };

export function scoreSegments(rows, maximumGap = 450000) {
  const segments = [];
  rows.forEach((row, index) => {
    if (!index || Date.parse(row.t) - Date.parse(rows[index - 1].t) > maximumGap) segments.push([]);
    segments.at(-1).push(row);
  });
  return segments;
}

export function describeEvent(event) {
  if (event.status === 'warned') return {
    title: 'A warning appeared before this fault.',
    body: `${event.alert_points} reading${event.alert_points === 1 ? '' : 's'} crossed the threshold in the preceding six hours. This is a timing association, not proof that the model identified the failure. A single crossing is weaker evidence than a sustained warning.`,
  };
  if (event.status === 'missed') return {
    title: 'The fault had no preceding warning.',
    body: 'Enough of the warning window was scored to assess this event, but no reading crossed the frozen threshold. The model missed this recorded fault under the declared six-hour rule.',
  };
  return {
    title: 'We cannot assess this fault fairly.',
    body: event.scored_points === 0
      ? 'There are no eligible model scores in the preceding six hours. Missing scores do not mean normal operation, and this event is not counted as a detection or a miss.'
      : `Some scores are available, but the coverage or latest-reading requirement is not met.${event.alert_points ? ' Threshold crossings are visible, but this event is excluded from the adequately scored result.' : ''} This event is not counted as a detection or a miss.`,
  };
}

export function validateEvidence(data) {
  if (data?.schema_version !== 1 || !Array.isArray(data.scores) || !data.scores.length || !Array.isArray(data.events) || !data.events.length || !data.summary || !data.provenance || typeof data.model !== 'string' || !(data.threshold > 0)) throw new Error('Unrecognized evidence format. Regenerate the local evidence file.');
  let previous = -Infinity;
  for (const row of data.scores) {
    const time = Date.parse(row.t);
    if (!Number.isFinite(time) || time <= previous || !Number.isFinite(row.ratio) || row.ratio < 0 || typeof row.alert !== 'boolean' || row.alert !== (row.ratio > 1)) throw new Error('Invalid saved score evidence.');
    previous = time;
  }
  for (const event of data.events) {
    if (!Object.hasOwn(LABELS, event.status) || !Number.isFinite(Date.parse(event.fault)) || !Number.isInteger(event.id) || !Number.isFinite(event.coverage) || event.coverage < 0 || event.coverage > 1) throw new Error('Invalid saved fault evidence.');
  }
  return data;
}
