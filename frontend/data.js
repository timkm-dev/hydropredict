export const CHANNELS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

export function demoTelemetry() {
  const start = Date.UTC(2026, 9, 18);
  return Array.from({ length: 288 }, (_, index) => {
    const row = { t: new Date(start + index * 300000).toISOString() };
    CHANNELS.forEach((channel, c) => {
      const wave = Math.sin(index / 13 + c) * (1.4 + c * 0.4);
      const ripple = Math.sin(index * 1.7 + c * 3) * 0.45;
      const excursion = index > 178 && index < 198 ? Math.sin((index - 178) / 20 * Math.PI) * (c === 0 ? 8 : 1.8) : 0;
      row[channel] = Number((24 + c * 13 + wave + ripple + excursion).toFixed(3));
    });
    return row;
  });
}

// RFC-style quoted fields, including escaped quotes and embedded newlines.
function csvRows(text) {
  const rows = [];
  let row = [], field = '', quoted = false, closed = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (char === '"') { quoted = false; closed = true; }
      else field += char;
    } else if (char === '"' && !field && !closed) quoted = true;
    else if (char === ',' || char === '\n' || char === '\r') {
      row.push(field); field = ''; closed = false;
      if (char !== ',') {
        if (row.some(value => value.trim())) rows.push(row);
        row = [];
        if (char === '\r' && text[i + 1] === '\n') i++;
      }
    } else {
      if (closed || char === '"') throw new Error('Malformed CSV quoting.');
      field += char;
    }
  }
  if (quoted) throw new Error('CSV contains an unclosed quoted field.');
  row.push(field);
  if (row.some(value => value.trim())) rows.push(row);
  return rows;
}

function timestamp(value, row) {
  // Explicit ISO dates avoid browser-dependent locale parsing. Naive times follow
  // the Python project's UTC contract; source timezone remains unverified.
  if (!/^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:Z|[+-]\d{2}:?\d{2})?)?$/.test(value)) {
    throw new Error(`Row ${row}: t must be an ISO date/time, for example 2019-01-01T00:00:00Z.`);
  }
  const dateOnly = value.slice(0, 10);
  const checkDate = new Date(`${dateOnly}T00:00:00Z`);
  if (!Number.isFinite(checkDate.getTime()) || checkDate.toISOString().slice(0, 10) !== dateOnly) {
    throw new Error(`Row ${row}: invalid calendar date.`);
  }
  const normalized = value.length === 10 ? `${value}T00:00:00Z` : value.replace(' ', 'T');
  const zoned = /(?:Z|[+-]\d{2}:?\d{2})$/.test(normalized) ? normalized : `${normalized}Z`;
  const date = new Date(zoned);
  if (!Number.isFinite(date.getTime()) || /T24:/.test(zoned)) throw new Error(`Row ${row}: invalid timestamp.`);
  return date.toISOString();
}

export function parseTelemetry(text) {
  const rows = csvRows(text.replace(/^\uFEFF/, ''));
  if (rows.length < 2) throw new Error('Include a header and at least one telemetry observation.');
  const headers = rows[0].map(value => value.trim());
  if (new Set(headers).size !== headers.length) throw new Error('CSV headers must be unique.');
  const columns = ['t', ...CHANNELS];
  if (columns.some(column => !headers.includes(column))) throw new Error('Required CSV columns: t,V1,V2,V3,V4,V5,V6.');
  if (rows.length > 100001) throw new Error('Import up to 100,000 observations at a time.');
  const seen = new Set();
  const records = rows.slice(1).map((values, index) => {
    const line = index + 2;
    if (values.length !== headers.length) throw new Error(`Row ${line}: field count does not match the header.`);
    const record = { t: timestamp(values[headers.indexOf('t')].trim(), line) };
    if (seen.has(record.t)) throw new Error(`Row ${line}: duplicate timestamp. Resolve duplicates before importing.`);
    seen.add(record.t);
    CHANNELS.forEach(channel => {
      const raw = values[headers.indexOf(channel)].trim();
      if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(raw) || !Number.isFinite(Number(raw))) {
        throw new Error(`Row ${line}: ${channel} must be a finite number.`);
      }
      record[channel] = Number(raw);
    });
    return record;
  });
  return records.sort((a, b) => Date.parse(a.t) - Date.parse(b.t));
}

export function windowRows(records, hours) {
  if (hours === 'all' || !records.length) return records;
  const cutoff = Date.parse(records.at(-1).t) - Number(hours) * 3600000;
  return records.filter(row => Date.parse(row.t) > cutoff);
}

export function toCSV(records) {
  return ['t,' + CHANNELS.join(','), ...records.map(row => [row.t, ...CHANNELS.map(c => row[c])].join(','))].join('\r\n');
}
