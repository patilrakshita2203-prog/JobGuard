import time

import pytest

from src.auth import store as auth
from src.auth.store import AuthError, UserStore


@pytest.fixture()
def users(tmp_path):
    return UserStore(tmp_path / "users.db")


def test_register_then_sign_in(users):
    user = users.register("Asha Kulkarni", "Asha@Example.com", "correct horse 9")
    assert user.email == "asha@example.com" and user.provider == "password"
    assert users.authenticate("  ASHA@example.com ", "correct horse 9").id == user.id


def test_password_is_never_stored_in_plain_text(users, tmp_path):
    users.register("Asha K", "asha@example.com", "correct horse 9")
    raw = (tmp_path / "users.db").read_bytes()
    assert b"correct horse 9" not in raw
    assert b"scrypt$" in raw


def test_same_password_gets_a_different_hash_each_time():
    assert auth.hash_password("same password 1") != auth.hash_password("same password 1")


@pytest.mark.parametrize("name,email,password,message", [
    ("A", "a@example.com", "longenough1", "name"),
    ("Asha", "not-an-email", "longenough1", "valid email"),
    ("Asha", "a@example.com", "short", "at least 8"),
    ("Asha", "a@example.com", "x" * 200, "at most"),
])
def test_registration_rules(users, name, email, password, message):
    with pytest.raises(AuthError, match=message):
        users.register(name, email, password)


def test_duplicate_email_is_rejected_case_insensitively(users):
    users.register("Asha", "asha@example.com", "longenough1")
    with pytest.raises(AuthError, match="already exists"):
        users.register("Other", "ASHA@EXAMPLE.COM", "longenough2")


def test_wrong_password_and_unknown_email_give_the_same_message(users):
    users.register("Asha", "asha@example.com", "longenough1")
    with pytest.raises(AuthError) as wrong:
        users.authenticate("asha@example.com", "nope-nope")
    with pytest.raises(AuthError) as unknown:
        users.authenticate("ghost@example.com", "nope-nope")
    assert str(wrong.value) == str(unknown.value) == "Incorrect email or password."


def test_account_locks_after_repeated_failures_and_recovers(users, monkeypatch):
    users.register("Asha", "asha@example.com", "longenough1")
    for _ in range(auth.MAX_FAILED_LOGINS):
        with pytest.raises(AuthError, match="Incorrect"):
            users.authenticate("asha@example.com", "wrong-pass")
    with pytest.raises(AuthError, match="Too many"):
        users.authenticate("asha@example.com", "longenough1")   # even the right password is refused

    later = time.time() + auth.LOCKOUT_SECONDS + 5
    monkeypatch.setattr(auth.time, "time", lambda: later)
    assert users.authenticate("asha@example.com", "longenough1").email == "asha@example.com"


def test_successful_sign_in_clears_earlier_failures(users):
    users.register("Asha", "asha@example.com", "longenough1")
    for _ in range(auth.MAX_FAILED_LOGINS - 1):
        with pytest.raises(AuthError):
            users.authenticate("asha@example.com", "wrong-pass")
    users.authenticate("asha@example.com", "longenough1")
    with pytest.raises(AuthError, match="Incorrect"):   # counter restarted, so not locked yet
        users.authenticate("asha@example.com", "wrong-pass")


def test_sessions_create_expire_and_delete(users, monkeypatch):
    user = users.register("Asha", "asha@example.com", "longenough1")
    token = users.create_session(user.id, days=30)
    assert users.user_for_session(token).email == "asha@example.com"
    assert users.user_for_session("not-a-real-token") is None
    assert users.user_for_session(None) is None

    users.delete_session(token)
    assert users.user_for_session(token) is None

    token = users.create_session(user.id, days=1)
    real_time = time.time
    monkeypatch.setattr(auth.time, "time", lambda: real_time() + 2 * 86400)
    assert users.user_for_session(token) is None


def test_garbage_session_values_are_ignored(users):
    for junk in (object(), 123, b"bytes", ["x"], {}):
        assert users.user_for_session(junk) is None
        users.delete_session(junk)   # must not raise


def test_only_a_hash_of_the_session_token_is_stored(users, tmp_path):
    user = users.register("Asha", "asha@example.com", "longenough1")
    token = users.create_session(user.id, days=30)
    assert token.encode() not in (tmp_path / "users.db").read_bytes()


def test_google_users_are_created_once_and_cannot_use_a_password(users):
    first = users.upsert_google_user("Gina@Example.com", "Gina G")
    again = users.upsert_google_user("gina@example.com", "Different Name")
    assert first.id == again.id and first.provider == "google"
    with pytest.raises(AuthError, match="Google"):
        users.authenticate("gina@example.com", "anything-at-all")


def test_google_sign_in_reuses_an_existing_password_account(users):
    existing = users.register("Asha", "asha@example.com", "longenough1")
    assert users.upsert_google_user("asha@example.com", "Asha").id == existing.id
