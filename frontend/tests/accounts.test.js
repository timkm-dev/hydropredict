import test from 'node:test';
import assert from 'node:assert/strict';
import {
  accountCounts, displayName, generatePassword, matchesAccount, showForRole, sinceText,
} from '../accounts.js';

const accounts = [
  { name: 'Ada Admin', email: 'ada@example.com', username: 'ada', company: 'Hydro', role: 'admin', disabled: false, sessions: 1 },
  { name: 'Ml Person', email: 'ml@example.com', username: 'ml', company: 'Wind', role: 'ml_engineer', disabled: false, sessions: 0 },
  { name: '', email: 'off@example.com', username: null, company: '', role: 'admin', disabled: true, sessions: 0 },
];

test('counts only active administrators and signed-in accounts', () => {
  assert.deepEqual(accountCounts(accounts), { total: 3, signedIn: 1, admins: 1, disabled: 1 });
});

test('filters by role, turned-off state and every search word', () => {
  assert.equal(accounts.filter(a => matchesAccount(a, '', 'admin')).length, 2);
  assert.deepEqual(accounts.filter(a => matchesAccount(a, '', 'disabled')).map(a => a.email), ['off@example.com']);
  assert.deepEqual(accounts.filter(a => matchesAccount(a, 'WIND ml')).map(a => a.username), ['ml']);
  assert.equal(accounts.filter(a => matchesAccount(a, 'hydro', 'ml_engineer')).length, 0);
});

test('names fall back to username then email', () => {
  assert.equal(displayName(accounts[0]), 'Ada Admin');
  assert.equal(displayName(accounts[2]), 'off@example.com');
  assert.equal(displayName(undefined), 'Deleted account');
});

test('describes how long ago something happened', () => {
  const now = Date.UTC(2026, 10, 4, 12);
  assert.equal(sinceText(null, now), 'Never');
  assert.equal(sinceText(now / 1000 - 20, now), 'Just now');
  assert.equal(sinceText(now / 1000 - 600, now), '10 min ago');
  assert.equal(sinceText(now / 1000 - 5 * 3600, now), '5 h ago');
  assert.equal(sinceText(now / 1000 - 3 * 86400, now), '3 days ago');
  assert.match(sinceText(now / 1000 - 40 * 86400, now), /Sept? 2026/);
});

test('generated passwords meet the length rule and are URL safe', () => {
  const password = generatePassword(new Uint8Array(18).fill(251));
  assert.equal(password.length, 24);
  assert.match(password, /^[A-Za-z0-9_-]+$/);
  assert.notEqual(generatePassword(), generatePassword());
});

test('shows links to every listed role', () => {
  const nodes = [{ dataset: { role: 'ml_engineer admin' } }, { dataset: { role: 'admin' } }];
  showForRole(nodes, 'ml_engineer');
  assert.deepEqual(nodes.map(n => n.hidden), [false, true]);
  showForRole(nodes, 'admin');
  assert.deepEqual(nodes.map(n => n.hidden), [false, false]);
});
