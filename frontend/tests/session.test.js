import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../session.js', import.meta.url), 'utf8');

async function renderSession({ connected = false, enabled = true } = {}) {
  const elements = new Map();
  const element = () => ({ hidden: false, textContent: '', style: {}, append() {}, replaceChildren() {} });
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
    querySelector: () => element(),
    createElement: () => element(),
  };
  const fetch = async url => ({
    status: 200, ok: true,
    headers: { get: () => 'application/json' },
    json: async () => url.endsWith('/config') ? { google_enabled: enabled } : {
      user: { name: 'Test Person', username: 'tester', company: 'Hydro',
        job_position: 'Engineer', google_connected: connected },
    },
  });
  await vm.runInNewContext(`(async () => { ${source}\n })()`, {
    document, fetch, location: { replace() { throw new Error('Unexpected redirect'); } },
  });
  return elements;
}

test('homepage shows Connect Google for an unlinked signed-in account', async () => {
  const elements = await renderSession();
  const button = elements.get('header-google');
  assert.equal(button.hidden, false);
  assert.equal(button.textContent, 'Connect Google');
  assert.equal(button.href, '/api/auth/google/start?link=1');
  assert.equal(elements.get('header-signout').hidden, false);
});

test('homepage identifies an already connected Google account', async () => {
  const button = (await renderSession({ connected: true })).get('header-google');
  assert.equal(button.hidden, false);
  assert.equal(button.textContent, 'Google connected');
  assert.equal(button.href, '/auth.html?profile');
});

test('homepage hides the Google action only when the server disables Google login', async () => {
  const button = (await renderSession({ enabled: false })).get('header-google');
  assert.equal(button.hidden, true);
});
