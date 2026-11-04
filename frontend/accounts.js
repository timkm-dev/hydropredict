// Account helpers shared by the page headers and the administration page (issue #14).
export const ROLE_LABEL = {
  operator: 'Maintenance Operator',
  ml_engineer: 'ML Engineer',
  admin: 'Administrator',
};
export const ROLE_BADGE = { operator: 'Operator', ml_engineer: 'ML Engineer', admin: 'Admin' };
export const ROLE_HELP = {
  operator: 'What to check and alert decisions',
  ml_engineer: 'Also the models and research pages',
  admin: 'Full access, including managing accounts',
};

// Header links carry data-role="ml_engineer admin" (space-separated) to say who sees them.
export function showForRole(nodes, role) {
  for (const node of nodes) node.hidden = !node.dataset.role.split(' ').includes(role);
}

export function accountCounts(accounts) {
  return {
    total: accounts.length,
    signedIn: accounts.filter(a => a.sessions > 0).length,
    admins: accounts.filter(a => a.role === 'admin' && !a.disabled).length,
    disabled: accounts.filter(a => a.disabled).length,
  };
}

// `show` is one of all, admin, ml_engineer, operator or disabled; the query matches text fields.
export function matchesAccount(account, query = '', show = 'all') {
  if (show === 'disabled' ? !account.disabled : show !== 'all' && account.role !== show) return false;
  const text = [account.name, account.email, account.username, account.company].join(' ').toLowerCase();
  return query.trim().toLowerCase().split(/\s+/).every(word => text.includes(word));
}

export function displayName(account) {
  return account?.name || account?.username || account?.email || 'Deleted account';
}

export function sinceText(seconds, now = Date.now()) {
  if (!seconds) return 'Never';
  const minutes = Math.round((now - seconds * 1000) / 60000);
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 48 * 60) return `${Math.round(minutes / 60)} h ago`;
  if (minutes < 14 * 24 * 60) return `${Math.round(minutes / 1440)} days ago`;
  return new Date(seconds * 1000).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}

// 18 random bytes as base64url: 24 characters, inside the 12–128 password rule.
export function generatePassword(bytes = crypto.getRandomValues(new Uint8Array(18))) {
  return btoa(String.fromCharCode(...bytes)).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
}
