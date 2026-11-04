import {
  ROLE_HELP, ROLE_LABEL, accountCounts, displayName, generatePassword, matchesAccount, sinceText,
} from './accounts.js';
import { ACTION_LABEL, formatMoment } from './care-model.js';

const $ = id => document.getElementById(id);
let accounts = [], me = null, show = 'all', selected = null;

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === 'class') node.className = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (key in node && key !== 'type') node[key] = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.filter(c => c !== null && c !== undefined && c !== false));
  return node;
}
function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
const ROLE_PILL = { admin: 'alarm', ml_engineer: 'watch', operator: 'normal' };
function rolePill(role) { return el('span', { class: `pill ${ROLE_PILL[role]}` }, ROLE_LABEL[role]); }
function dateText(seconds) {
  return seconds ? new Date(seconds * 1000).toLocaleString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : 'Not recorded';
}

async function api(path, body) {
  const response = await fetch(path, body === undefined
    ? { cache: 'no-store' }
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (response.status === 401) { location.assign('/auth.html'); throw new Error('Please sign in.'); }
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed. Try again.');
  return result;
}

function roleOptions(select) {
  select.replaceChildren(...Object.entries(ROLE_LABEL).map(([value, label]) => el('option', { value }, label)));
}

function renderSummary() {
  const counts = accountCounts(accounts);
  $('count-total').textContent = counts.total;
  $('count-live').textContent = counts.signedIn;
  $('count-admins').textContent = counts.admins;
  $('count-off').textContent = counts.disabled;
}

function renderAccounts() {
  const shown = accounts.filter(a => matchesAccount(a, $('search').value, show));
  $('shown-count').textContent = `Showing ${shown.length} of ${accounts.length}`;
  if (!shown.length) {
    $('account-rows').replaceChildren(el('p', { class: 'empty' }, 'No accounts match.'));
    return;
  }
  $('account-rows').replaceChildren(...shown.map(account => el('button', { type: 'button', class: 'row', onclick: () => openManage(account.id) },
    el('span', { class: 'who' },
      el('strong', {}, displayName(account), account.id === me ? ' (you)' : ''),
      el('small', {}, account.username ? `${account.email} · @${account.username}` : account.email)),
    el('span', { class: 'tags' },
      rolePill(account.role),
      account.disabled ? el('span', { class: 'pill off' }, 'Turned off') : null,
      !account.username && !account.disabled ? el('span', { class: 'pill done' }, 'Profile not finished') : null,
      account.sessions > 0 && !account.disabled ? el('span', { class: 'live' }, 'Signed in') : null),
    el('span', { class: 'when' }, `Last sign-in: ${sinceText(account.last_login)}`),
    el('span', { class: 'open' }, 'Manage'))));
}

function render() {
  renderSummary();
  renderAccounts();
  if (selected !== null && $('manage').open) fillManage();
}

function setAccounts(result) {
  accounts = result.accounts;
  me = result.me;
  render();
}

async function renderActivity() {
  const { reviews } = await api('/api/admin/activity');
  const byId = new Map(accounts.map(a => [String(a.id), a]));
  if (!reviews.length) {
    $('activity-rows').replaceChildren(el('p', { class: 'empty' }, 'No alert decisions yet.'));
    return;
  }
  $('activity-rows').replaceChildren(...reviews.slice(0, 30).map(review => el('a', { class: 'row', href: `care.html#/alerts/${encodeURIComponent(review.alert)}` },
    el('span', { class: 'who' }, el('strong', {}, displayName(byId.get(review.owner))), el('small', {}, `Alert ${review.alert}`)),
    el('span', { class: 'tags' }, el('span', { class: `pill ${review.action === 'inspect' ? 'alarm' : review.action === 'open' ? 'watch' : 'done'}` }, ACTION_LABEL[review.action] ?? review.action)),
    el('span', { class: 'text' }, review.comment ? `“${review.comment}”` : ''),
    el('span', { class: 'when' }, formatMoment(review.created)))));
}

// Account dialog
function current() { return accounts.find(a => a.id === selected); }

function fillManage() {
  const account = current();
  if (!account) { $('manage').close(); return; }
  const self = account.id === me;
  $('manage-name').textContent = displayName(account);
  $('manage-email').textContent = account.email;
  $('manage-self').hidden = !self;
  const methods = [account.has_password ? 'Password' : null, account.google_connected ? 'Google' : null].filter(Boolean);
  const facts = [
    ['Username', account.username ? `@${account.username}` : 'Profile not finished'],
    ['Company', account.company || '—'],
    ['Job position', account.job_position || '—'],
    ['Signs in with', methods.join(' and ') || '—'],
    ['Account created', dateText(account.created)],
    ['Last sign-in', account.last_login ? dateText(account.last_login) : 'Never'],
    ['Open sessions', String(account.sessions)],
  ];
  $('manage-facts').replaceChildren(...facts.flatMap(([term, value]) => [el('dt', {}, term), el('dd', {}, value)]));
  $('manage-role').value = account.role;
  $('manage-role').disabled = self;
  $('manage-role-save').disabled = self;
  $('manage-access-text').textContent = account.disabled
    ? 'This account is turned off: it cannot sign in. Turning it back on lets them sign in with their existing password.'
    : 'Signing out ends every open session. Turning the account off also stops them signing in again.';
  $('manage-toggle').textContent = account.disabled ? 'Turn account on' : 'Turn account off';
  $('manage-toggle').disabled = self;
  $('manage-confirm-hint').textContent = account.email;
  $('manage-confirm').disabled = self;
  $('manage-delete').disabled = self || $('manage-confirm').value.trim().toLowerCase() !== account.email;
}

function openManage(id) {
  selected = id;
  $('dialog-message').textContent = '';
  $('manage-confirm').value = '';
  $('manage-password').value = generatePassword();
  fillManage();
  $('manage').showModal();
}

async function act(action, extra = {}, done = '') {
  const account = current();
  $('dialog-message').textContent = '';
  try {
    setAccounts(await api('/api/admin/account', { id: account.id, action, ...extra }));
    if (action === 'delete') {
      $('manage').close();
      message(`Deleted ${account.email}.`);
      await renderActivity();
    } else {
      $('dialog-message').textContent = done;
    }
  } catch (error) {
    $('dialog-message').textContent = error.message;
  }
}

$('manage-close').onclick = () => $('manage').close();
$('manage').addEventListener('click', event => { if (event.target === $('manage')) $('manage').close(); });
$('manage-generate').onclick = () => { $('manage-password').value = generatePassword(); };
$('manage-role-save').onclick = () => act('role', { role: $('manage-role').value }, `Role saved: ${ROLE_LABEL[$('manage-role').value]}. It applies on their next page load.`);
$('manage-password-save').onclick = () => {
  const password = $('manage-password').value;
  act('password', { password }, `Password reset to ${password} and every session signed out. Share it with them privately.`);
};
$('manage-signout').onclick = () => act('sign_out', {}, 'Signed out of every session.');
$('manage-toggle').onclick = () => {
  const off = current().disabled;
  act(off ? 'enable' : 'disable', {}, off ? 'Account turned on. They can sign in again.' : 'Account turned off and signed out.');
};
$('manage-confirm').oninput = fillManage;
$('manage-delete').onclick = () => act('delete');

// Add an account
$('create-generate').onclick = () => { $('create-password').value = generatePassword(); };
$('create-form').onsubmit = async event => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target));
  const button = event.target.querySelector('button.btn:not(.plain)');
  button.disabled = true;
  $('create-error').textContent = '';
  try {
    setAccounts(await api('/api/admin/create', data));
    $('handover-email').textContent = data.email.trim().toLowerCase();
    $('handover-password').textContent = data.password;
    $('handover').hidden = false;
    event.target.reset();
    $('create-password').value = generatePassword();
  } catch (error) {
    $('create-error').textContent = error.message;
  } finally {
    button.disabled = false;
  }
};
$('handover-copy').onclick = async () => {
  const text = `Early Warning sign-in\n${location.origin}/auth.html\nEmail: ${$('handover-email').textContent}\nPassword: ${$('handover-password').textContent}`;
  try { await navigator.clipboard.writeText(text); $('handover-copy').textContent = 'Copied'; }
  catch { $('handover-copy').textContent = 'Copy blocked: select the text instead'; }
  setTimeout(() => { $('handover-copy').textContent = 'Copy sign-in details'; }, 2500);
};

// Filters
$('search').oninput = renderAccounts;
for (const chip of document.querySelectorAll('[data-show]')) {
  chip.onclick = () => {
    show = chip.dataset.show;
    for (const other of document.querySelectorAll('[data-show]')) other.setAttribute('aria-pressed', String(other === chip));
    renderAccounts();
  };
}

roleOptions($('create-role'));
roleOptions($('manage-role'));
$('role-help').replaceChildren(...Object.entries(ROLE_HELP).flatMap(([role, help]) => [el('dt', {}, ROLE_LABEL[role]), el('dd', {}, help)]));
$('create-password').value = generatePassword();
try {
  setAccounts(await api('/api/admin/accounts'));
  await renderActivity();
} catch (error) {
  message(error.message, true);
}
