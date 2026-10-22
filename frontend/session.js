const message = document.getElementById('session-message');
async function logout(button) {
  button.disabled = true;
  message.textContent = '';
  try {
    const response = await fetch('/api/auth/logout', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
    });
    if (!response.ok) throw new Error('Sign out failed. Please try again.');
    location.replace('/auth.html');
  } catch (error) {
    message.textContent = error.message;
  } finally { button.disabled = false; }
}
const headerLogout = document.getElementById('header-signout');
const googleButton = document.getElementById('header-google');
headerLogout.onclick = () => logout(headerLogout);
try {
  const response = await fetch('/api/auth/me', { cache: 'no-store' });
  if (response.status === 401) location.replace('/auth.html');
  else if (response.ok && response.headers.get('content-type')?.includes('application/json')) {
    const { user } = await response.json();
    if (!user.username) location.replace('/auth.html');
    document.getElementById('header-signin').hidden = true;
    document.getElementById('header-register').hidden = true;
    const headerProfile = document.getElementById('header-profile');
    headerProfile.hidden = false;
    headerProfile.textContent = user.name || 'My profile';
    headerLogout.hidden = false;
    googleButton.textContent = user.google_connected ? 'Google connected' : 'Connect Google';
    googleButton.href = user.google_connected ? '/auth.html?profile' : '/api/auth/google/start?link=1';
    const profile = document.querySelector('.profile');
    profile.replaceChildren();
    const details = document.createElement('a');
    details.href = '/auth.html?profile';
    details.style.cssText = 'color:inherit;text-decoration:none;flex:1';
    details.textContent = user.name;
    const subtitle = document.createElement('small');
    subtitle.textContent = `${user.job_position} · ${user.company}`;
    details.append(subtitle);
    const logout = document.createElement('button');
    logout.className = 'text-button';
    logout.textContent = 'Sign out';
    logout.onclick = async () => {
      const result = await fetch('/api/auth/logout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      if (result.ok) location.replace('/auth.html');
      else { logout.textContent = 'Retry sign out'; }
    };
    profile.append(details, logout);
  } else {
    message.textContent = 'Accounts require the dashboard server.';
  }
} catch { message.textContent = 'Account connection unavailable. Please try again.'; }

try {
  const response = await fetch('/api/auth/config', { cache: 'no-store' });
  const config = await response.json();
  googleButton.hidden = !config.google_enabled;
} catch { /* Keep the Google entry point visible; its endpoint reports availability. */ }
