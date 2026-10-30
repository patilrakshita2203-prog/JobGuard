"""The login page, driven through the real dashboard script (no browser, no network)."""

import pytest

import src.auth.login_page as login_page
from conftest import ROOT, needs_models
from src import config as cfg
from src.auth.store import UserStore

pytestmark = needs_models

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture()
def protected(monkeypatch, tmp_path):
    """Turn the login on with a fresh, empty user database."""
    monkeypatch.setattr(cfg, "AUTH_REQUIRED", True)
    monkeypatch.setattr(cfg, "AUTH_DB_PATH", tmp_path / "users.db")
    monkeypatch.setattr(login_page, "_cookie_token", lambda: None)
    return UserStore(tmp_path / "users.db")


def open_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()


def submit(at, label):
    return [b for b in at.button if b.label == label][0].click().run()


def sign_up(at, name="Asha K", email="asha@example.com", password="longenough1", confirm=None):
    at.text_input(key="signup_name").input(name)
    at.text_input(key="signup_email").input(email)
    at.text_input(key="signup_password").input(password)
    at.text_input(key="signup_confirm").input(confirm or password)
    return submit(at, "Create account")


def test_nothing_is_visible_before_signing_in(protected):
    at = open_app()
    assert not at.exception
    assert at.text_input(key="signin_email")
    assert len(at.sidebar.radio) == 0          # no menu
    assert "Single Analyzer" not in " ".join(m.value for m in at.markdown)


def test_google_button_is_disabled_until_google_keys_exist(protected):
    at = open_app()
    assert at.button(key="google_login").disabled is True


def test_create_account_signs_you_in_and_shows_the_app(protected):
    at = sign_up(open_app())
    assert not at.exception
    assert len(at.sidebar.radio(key="nav_page").options) == 5
    assert "Asha K" in " ".join(m.value for m in at.sidebar.markdown)
    assert protected.authenticate("asha@example.com", "longenough1").name == "Asha K"


def test_mismatched_passwords_are_refused(protected):
    at = sign_up(open_app(), confirm="different-pass-1")
    assert any("do not match" in e.value for e in at.error)
    assert len(at.sidebar.radio) == 0


def test_weak_password_message_is_shown(protected):
    at = sign_up(open_app(), password="short")
    assert any("at least 8" in e.value for e in at.error)


def test_sign_in_with_wrong_then_right_password(protected):
    protected.register("Asha K", "asha@example.com", "longenough1")
    at = open_app()
    at.text_input(key="signin_email").input("asha@example.com")
    at.text_input(key="signin_password").input("wrong-password")
    at = submit(at, "Sign in")
    assert any("Incorrect email or password" in e.value for e in at.error)

    at.text_input(key="signin_password").input("longenough1")
    at = submit(at, "Sign in")
    assert not at.exception and len(at.sidebar.radio) == 1


def test_five_wrong_passwords_lock_the_account(protected):
    protected.register("Asha K", "asha@example.com", "longenough1")
    at = open_app()
    at.text_input(key="signin_email").input("asha@example.com")
    for _ in range(5):
        at.text_input(key="signin_password").input("wrong-password")
        at = submit(at, "Sign in")
    at.text_input(key="signin_password").input("longenough1")
    at = submit(at, "Sign in")
    assert any("Too many failed attempts" in e.value for e in at.error)


def test_sign_out_returns_to_the_login_page(protected):
    at = sign_up(open_app())
    at = at.button(key="sign_out").click().run()
    assert at.text_input(key="signin_email")          # the login form is back
    assert "auth_user" not in at.session_state        # and nobody is signed in
    # (the sidebar is hidden by the page; a real browser clears it, the test runner keeps stale items)


def test_a_saved_session_cookie_keeps_you_signed_in_after_a_refresh(protected, monkeypatch):
    user = protected.register("Asha K", "asha@example.com", "longenough1")
    token = protected.create_session(user.id, days=30)
    monkeypatch.setattr(login_page, "_cookie_token", lambda: token)
    at = open_app()
    assert not at.exception
    assert len(at.sidebar.radio(key="nav_page").options) == 5


def test_an_expired_or_unknown_cookie_does_not_sign_you_in(protected, monkeypatch):
    monkeypatch.setattr(login_page, "_cookie_token", lambda: "forged-token")
    assert len(open_app().sidebar.radio) == 0


def test_returning_from_google_signs_you_in_and_creates_the_account(protected, monkeypatch):
    monkeypatch.setattr(login_page, "_google_identity", lambda: ("gina@example.com", "Gina G"))
    at = open_app()
    assert not at.exception and len(at.sidebar.radio) == 1
    assert protected.upsert_google_user("gina@example.com", "x").provider == "google"


def test_login_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(cfg, "AUTH_REQUIRED", False)
    at = open_app()
    assert len(at.sidebar.radio) == 1


def test_google_button_starts_the_google_login_when_keys_exist(protected, monkeypatch):
    import streamlit as st

    started = []
    monkeypatch.setattr(login_page, "google_configured", lambda: True)
    monkeypatch.setattr(st, "login", lambda *args, **kwargs: started.append(True))
    at = open_app()
    assert at.button(key="google_login").disabled is False
    at.button(key="google_login").click().run()
    assert started, "clicking the button should call st.login()"
