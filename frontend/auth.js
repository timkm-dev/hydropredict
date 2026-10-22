let mode = 'login';
const $ = id => document.getElementById(id);
async function request(action, data) {
  const response = await fetch(`/api/auth/${action}`, {
    method: data === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    ...(data === undefined ? {} : { body: JSON.stringify(data) }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Please sign in to continue.');
  return result.user;
}
function showProfile(user) {
  $('google-login').hidden = true;
  if (user.google_connected) {
    $('connect-google').textContent = 'Google connected';
    $('connect-google').removeAttribute('href');
    $('connect-google').setAttribute('aria-disabled', 'true');
  }
  $('credentials-form').hidden = true;
  $('auth-tabs').hidden = true;
  $('profile-form').hidden = false;
  $('step-label').textContent = 'MAKE IT YOURS';
  $('auth-title').textContent = 'Complete your profile';
  $('auth-description').textContent = 'Tell us a little about yourself and your organization.';
  for (const key of ['name', 'company', 'username', 'job_position']) {
    $('profile-form').elements[key].value = user[key] || '';
  }
  $('name').focus();
}
function proceed(user) {
  if (user.username) location.replace('/');
  else showProfile(user);
}
function setMode(next) {
  mode = next;
  const register = mode === 'register';
  $('login-tab').setAttribute('aria-pressed', String(!register));
  $('register-tab').setAttribute('aria-pressed', String(register));
  $('auth-title').textContent = register ? 'Create your account' : 'Sign in to HydroPredict';
  $('step-label').textContent = register ? 'GET STARTED' : 'WELCOME BACK';
  $('auth-description').textContent = register ? 'First, set up your email and password.' : 'Enter your details to open your workspace.';
  $('credentials-submit').textContent = register ? 'Create account & continue' : 'Sign in';
  $('password').autocomplete = register ? 'new-password' : 'current-password';
  $('password').minLength = register ? 12 : 1;
  $('password-help').hidden = !register;
  $('confirm-field').hidden = !register;
  $('confirm-password').required = register;
  $('auth-message').textContent = '';
}
$('login-tab').onclick = () => setMode('login');
$('register-tab').onclick = () => setMode('register');
$('show-password').onclick = () => {
  const show = $('password').type === 'password';
  $('password').type = show ? 'text' : 'password';
  $('show-password').textContent = show ? 'Hide' : 'Show';
  $('show-password').setAttribute('aria-label', show ? 'Hide password' : 'Show password');
};
async function submit(form, action, data) {
  const button = form.querySelector('button[type="submit"], .auth-submit');
  button.disabled = true;
  $('auth-message').textContent = '';
  try { proceed(await request(action, data)); }
  catch (error) { $('auth-message').textContent = error.message; }
  finally { button.disabled = false; }
}
$('credentials-form').onsubmit = event => {
  event.preventDefault();
  if (mode === 'register' && $('password').value !== $('confirm-password').value) {
    $('auth-message').textContent = 'Passwords do not match.';
    return;
  }
  submit(event.target, mode, Object.fromEntries(new FormData(event.target)));
};
$('profile-form').onsubmit = event => {
  event.preventDefault();
  submit(event.target, 'profile', Object.fromEntries(new FormData(event.target)));
};
$('profile-signout').onclick = async () => {
  try { await request('logout', {}); location.reload(); }
  catch (error) { $('auth-message').textContent = error.message; }
};
if (new URLSearchParams(location.search).get('mode') === 'register') setMode('register');
const oauthError = new URLSearchParams(location.search).get('error');
if (oauthError) {
  $('auth-message').textContent = oauthError;
  history.replaceState(null, '', location.pathname);
}
try {
  const response = await fetch('/api/auth/config', { cache: 'no-store' });
  const config = await response.json();
  $('google-login').hidden = !config.google_enabled;
  $('connect-google').hidden = !config.google_enabled;
} catch { /* Email and password remain available. */ }
try {
  const user = await request('me');
  if (oauthError || new URLSearchParams(location.search).has('profile')) showProfile(user);
  else proceed(user);
} catch (error) {
  if (error instanceof TypeError) $('auth-message').textContent = 'Unable to connect. Start the HydroPredict server and try again.';
}
