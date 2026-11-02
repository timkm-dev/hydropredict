// Shared header behaviour for pages outside the main app: role badge, role links and sign-out.
const response = await fetch('/api/auth/me', { cache: 'no-store' });
if (response.status === 401) location.assign('/auth.html');
const { user } = response.ok ? await response.json() : { user: null };
if (user) {
  const role = document.getElementById('role');
  if (role) role.textContent = user.role === 'ml_engineer' ? 'ML Engineer' : 'Operator';
  for (const node of document.querySelectorAll('[data-role]')) node.hidden = node.dataset.role !== user.role;
}
document.getElementById('logout')?.addEventListener('click', async () => {
  await fetch('/api/auth/logout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  location.replace('/');
});
