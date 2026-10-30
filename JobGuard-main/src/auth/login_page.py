"""
Login page and session handling for the dashboard.

Sign in with:
  - Google (Streamlit's built-in OpenID Connect login; needs the [auth] secrets, see README)
  - email and password (accounts live in a local SQLite file, see store.py)

"Keep me signed in" stores a random session token in a browser cookie, so a page
refresh does not log you out. Only a hash of the token is kept on the server.
"""

import json

import streamlit as st
import streamlit.components.v1 as components

from src import config as cfg
from src.auth.store import AuthError, UserStore

COOKIE_NAME = "jobguard_session"

LOGIN_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.login-brand { font-size: 2.2rem; font-weight: 700; text-align: center; margin-top: 2.5rem; }
.login-sub { text-align: center; color: #cbd5e1; margin: 0.4rem 0 1.6rem; }
.login-or { display: flex; align-items: center; gap: 0.8rem; color: #94a3b8; font-size: 0.85rem; margin: 1.1rem 0 0.4rem; }
.login-or::before, .login-or::after { content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.15); }
/* readable labels, a calm disabled button, and a dark password field on the dark page */
[data-testid="stCheckbox"] p, [data-testid="stForm"] label p { color: #e2e8f0 !important; }
.stButton > button:disabled {
    background: rgba(255,255,255,0.05) !important; color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.15) !important; box-shadow: none !important;
}
div[data-baseweb="input"], div[data-baseweb="input"] > div { background-color: #1e293b !important; border-radius: 8px; }
div[data-baseweb="input"] button { background: transparent !important; }
div[data-baseweb="input"] svg { fill: #cbd5e1 !important; }
</style>
"""


@st.cache_resource
def get_store(db_path: str) -> UserStore:
    return UserStore(db_path)


# ───────── browser cookie (set/cleared with a tiny script, applied on the next run)

def _run_cookie_script(script: str):
    components.html(f"<script>{script}</script>", height=0)


def _apply_pending_cookie_changes():
    pending_set = st.session_state.pop("_cookie_set", None)
    if pending_set:
        token, max_age = pending_set
        max_age_part = f"; max-age={int(max_age)}" if max_age else ""
        _run_cookie_script(
            f"const c = {json.dumps(COOKIE_NAME)} + '=' + {json.dumps(token)} + '; path=/; SameSite=Lax{max_age_part}'"
            " + (window.location.protocol === 'https:' ? '; Secure' : '');"
            "try { window.parent.document.cookie = c; } catch (e) { document.cookie = c; }"
        )
    if st.session_state.pop("_cookie_clear", False):
        _run_cookie_script(
            f"const c = {json.dumps(COOKIE_NAME)} + '=; path=/; max-age=0; SameSite=Lax';"
            "try { window.parent.document.cookie = c; } catch (e) { document.cookie = c; }"
        )


def _cookie_token():
    try:
        value = st.context.cookies.get(COOKIE_NAME)
    except Exception:
        return None
    return value if isinstance(value, str) and value else None  # anything else is not a real cookie


# ───────── Google

def google_configured() -> bool:
    try:
        auth = st.secrets["auth"]
        return bool(auth.get("client_id")) and bool(auth.get("client_secret"))
    except Exception:
        return False


def _google_identity():
    """(email, name) when Streamlit's Google login has completed, else None."""
    try:
        user = st.user
        if user.get("is_logged_in") and user.get("email"):
            return user.get("email"), user.get("name") or ""
    except Exception:
        pass
    return None


# ───────── session

def current_user():
    """The signed-in user as a dict, restored from this session, Google or the cookie."""
    store = get_store(str(cfg.AUTH_DB_PATH))
    user = st.session_state.get("auth_user")
    if user:
        return user

    identity = _google_identity()
    if identity:
        user = store.upsert_google_user(*identity).as_dict()
        st.session_state["auth_user"] = user
        return user

    token = _cookie_token()
    found = store.user_for_session(token)
    if found:
        st.session_state["auth_user"] = found.as_dict()
        st.session_state["auth_token"] = token
        return st.session_state["auth_user"]
    return None


def sign_in(user, remember: bool):
    store = get_store(str(cfg.AUTH_DB_PATH))
    days = cfg.SESSION_DAYS if remember else 1
    token = store.create_session(user.id, days)
    st.session_state["auth_user"] = user.as_dict()
    st.session_state["auth_token"] = token
    st.session_state["_cookie_set"] = (token, days * 86400 if remember else None)


def sign_out():
    store = get_store(str(cfg.AUTH_DB_PATH))
    provider = (st.session_state.get("auth_user") or {}).get("provider")
    store.delete_session(st.session_state.get("auth_token") or _cookie_token())
    for key in ("auth_user", "auth_token"):
        st.session_state.pop(key, None)
    st.session_state["_cookie_clear"] = True
    # also forget the signed-in page and the last report
    for key in ("single", "bulk", "scan", "scan_ai"):
        st.session_state.pop(key, None)
    if provider == "google":
        try:
            st.logout()
        except Exception:
            pass


# ───────── the page

def render_login_page():
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)
    store = get_store(str(cfg.AUTH_DB_PATH))

    _, middle, _ = st.columns([1, 1.2, 1])
    with middle:
        st.markdown('<div class="login-brand">🛡️ JobGuard Pro</div>'
                    '<div class="login-sub">Sign in to check job postings for scams.</div>', unsafe_allow_html=True)

        if google_configured():
            st.button("Continue with Google", key="google_login", use_container_width=True, on_click=lambda: st.login())
        else:
            st.button("Continue with Google", key="google_login", use_container_width=True, disabled=True)
            st.caption("Google sign-in is not set up yet. Add your Google keys to turn it on (see the README).")

        st.markdown('<div class="login-or">or use your email</div>', unsafe_allow_html=True)

        labels = ["Sign in", "Create account"] if cfg.ALLOW_SIGNUP else ["Sign in"]
        tabs = st.tabs(labels)

        with tabs[0]:
            with st.form("signin_form"):
                email = st.text_input("Email", key="signin_email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", key="signin_password")
                remember = st.checkbox("Keep me signed in", value=True, key="signin_remember")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                try:
                    sign_in(store.authenticate(email, password), remember)
                    st.rerun()
                except AuthError as exc:
                    st.error(str(exc))

        if cfg.ALLOW_SIGNUP:
            with tabs[1]:
                with st.form("signup_form"):
                    name = st.text_input("Full name", key="signup_name")
                    email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
                    password = st.text_input("Password", type="password", key="signup_password",
                                             help="At least 8 characters.")
                    confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                    submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
                if submitted:
                    try:
                        if password != confirm:
                            raise AuthError("The two passwords do not match.")
                        sign_in(store.register(name, email, password), remember=True)
                        st.rerun()
                    except AuthError as exc:
                        st.error(str(exc))


def require_login() -> dict:
    """Return the signed-in user, or show the login page and stop the script."""
    if not cfg.AUTH_REQUIRED:
        return {"id": 0, "email": "", "name": "Guest", "provider": "none"}

    user = current_user()
    _apply_pending_cookie_changes()
    if user:
        return user

    render_login_page()
    st.stop()
