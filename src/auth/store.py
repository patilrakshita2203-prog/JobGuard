"""
User accounts and login sessions (SQLite, standard library only).

- Passwords are never stored: only a salted scrypt hash.
- Session tokens are random; only their SHA-256 hash is stored.
- After too many wrong passwords an account is locked for a few minutes.
- Google sign-in users are stored without a password.
"""

import base64
import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MIN_PASSWORD = 8
MAX_PASSWORD = 128
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 15 * 60

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}


class AuthError(Exception):
    """A problem the user can fix (shown as-is in the login form)."""


@dataclass
class User:
    id: int
    email: str
    name: str
    provider: str  # "password" or "google"

    def as_dict(self):
        return {"id": self.id, "email": self.email, "name": self.name, "provider": self.provider}


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return "scrypt${}${}${}${}${}".format(
        _SCRYPT["n"], _SCRYPT["r"], _SCRYPT["p"],
        base64.b64encode(salt).decode(), base64.b64encode(digest).decode(),
    )


def verify_password(password: str, stored: Optional[str]) -> bool:
    """Constant-time check. Always does the expensive hash, even for unknown accounts."""
    if not stored or not stored.startswith("scrypt$"):
        hashlib.scrypt(password.encode("utf-8"), salt=b"0" * 16, **_SCRYPT)  # dummy work: no timing leak
        return False
    try:
        _, n, r, p, salt_b64, digest_b64 = stored.split("$")
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(digest_b64)
        digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class UserStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    password_hash TEXT,
                    provider TEXT NOT NULL DEFAULT 'password',
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS failed_logins (
                    email TEXT NOT NULL,
                    at REAL NOT NULL
                );
                """
            )

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ───────── accounts
    def register(self, name: str, email: str, password: str) -> User:
        name, email = (name or "").strip(), normalize_email(email)
        if not 2 <= len(name) <= 60:
            raise AuthError("Enter your name (2 to 60 characters).")
        if not _EMAIL_RE.match(email) or len(email) > 254:
            raise AuthError("Enter a valid email address.")
        if len(password or "") < MIN_PASSWORD:
            raise AuthError(f"Use a password of at least {MIN_PASSWORD} characters.")
        if len(password) > MAX_PASSWORD:
            raise AuthError(f"The password can be at most {MAX_PASSWORD} characters.")

        password_hash = hash_password(password)
        try:
            with closing(self._connect()) as conn, conn:
                cur = conn.execute(
                    "INSERT INTO users (email, name, password_hash, provider, created_at) VALUES (?, ?, ?, 'password', ?)",
                    (email, name, password_hash, time.time()),
                )
                return User(cur.lastrowid, email, name, "password")
        except sqlite3.IntegrityError:
            raise AuthError("An account with this email already exists. Try signing in.")

    def authenticate(self, email: str, password: str) -> User:
        email = normalize_email(email)
        if self._is_locked(email):
            raise AuthError("Too many failed attempts. Please wait a few minutes and try again.")

        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id, email, name, password_hash, provider FROM users WHERE email = ?", (email,)
            ).fetchone()

        ok = verify_password(password or "", row[3] if row else None)
        if not ok or row is None:
            self._record_failure(email)
            if row is not None and row[3] is None and row[4] == "google":
                raise AuthError("This account uses Google sign-in. Use the Google button.")
            raise AuthError("Incorrect email or password.")

        self._clear_failures(email)
        return User(row[0], row[1], row[2], row[4])

    def upsert_google_user(self, email: str, name: str) -> User:
        """Google has already verified the email, so sign the person in (creating the account once)."""
        email = normalize_email(email)
        if not _EMAIL_RE.match(email):
            raise AuthError("Google did not return a valid email address.")
        name = (name or email.split("@")[0]).strip()[:60] or "User"
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT id, name, provider FROM users WHERE email = ?", (email,)).fetchone()
            if row:
                return User(row[0], email, row[1], row[2])
            cur = conn.execute(
                "INSERT INTO users (email, name, password_hash, provider, created_at) VALUES (?, ?, NULL, 'google', ?)",
                (email, name, time.time()),
            )
            return User(cur.lastrowid, email, name, "google")

    # ───────── lockout bookkeeping
    def _is_locked(self, email: str) -> bool:
        cutoff = time.time() - LOCKOUT_SECONDS
        with closing(self._connect()) as conn:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM failed_logins WHERE email = ? AND at > ?", (email, cutoff)
            ).fetchone()
        return count >= MAX_FAILED_LOGINS

    def _record_failure(self, email: str):
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT INTO failed_logins (email, at) VALUES (?, ?)", (email, time.time()))
            conn.execute("DELETE FROM failed_logins WHERE at < ?", (time.time() - LOCKOUT_SECONDS,))

    def _clear_failures(self, email: str):
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM failed_logins WHERE email = ?", (email,))

    # ───────── sessions ("keep me signed in")
    def create_session(self, user_id: int, days: float) -> str:
        token = secrets.token_urlsafe(32)
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                (_hash_token(token), user_id, time.time() + days * 86400),
            )
        return token

    def user_for_session(self, token: Optional[str]) -> Optional[User]:
        if not isinstance(token, str) or not token:
            return None
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT u.id, u.email, u.name, u.provider FROM sessions s JOIN users u ON u.id = s.user_id "
                "WHERE s.token_hash = ? AND s.expires_at > ?",
                (_hash_token(token), time.time()),
            ).fetchone()
        return User(*row) if row else None

    def delete_session(self, token: Optional[str]):
        if not isinstance(token, str) or not token:
            return
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))
