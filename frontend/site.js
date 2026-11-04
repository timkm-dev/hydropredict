// Shared header behaviour for pages outside the main app: role badge, role links and sign-out.
import { ROLE_BADGE, showForRole } from './accounts.js';

const response = await fetch('/api/auth/me', { cache: 'no-store' });
if (response.status === 401) location.assign('/auth.html');
const { user } = response.ok ? await response.json() : { user: null };
if (user) {
  const role = document.getElementById('role');
  if (role) role.textContent = ROLE_BADGE[user.role];
  showForRole(document.querySelectorAll('[data-role]'), user.role);
}
document.getElementById('logout')?.addEventListener('click', async () => {
  await fetch('/api/auth/logout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  location.replace('/');
});
