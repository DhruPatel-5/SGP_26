from datetime import datetime
import hashlib

import streamlit as st

from truthguard.config import DEFAULT_USERS
from truthguard.db import get_conn


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def ensure_default_users() -> None:
    with get_conn() as conn:
        for user in DEFAULT_USERS:
            exists = conn.execute("SELECT id FROM users WHERE username = ?", (user["username"],)).fetchone()
            if not exists:
                conn.execute(
                    """
                    INSERT INTO users (username, full_name, email, password_hash, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user["username"],
                        user["full_name"],
                        user["email"],
                        hash_password(user["password"]),
                        user["role"],
                        datetime.utcnow().isoformat(),
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE users
                    SET full_name = ?, email = ?, password_hash = ?, role = ?
                    WHERE username = ?
                    """,
                    (
                        user["full_name"],
                        user["email"],
                        hash_password(user["password"]),
                        user["role"],
                        user["username"],
                    ),
                )


def authenticate(username: str, password: str):
    with get_conn() as conn:
        user = conn.execute(
            "SELECT username, full_name, email, role, password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if not user:
        return None
    if user["password_hash"] != hash_password(password):
        return None
    return {
        "username": user["username"],
        "full_name": user["full_name"],
        "email": user["email"],
        "role": user["role"],
    }


def login_view() -> None:
    st.markdown("## TRUTHGUARD Login")
    st.caption("Demo users: admin / 123, celebrity / 123, analyst / 123")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
    if submitted:
        user = authenticate(username, password)
        if user:
            st.session_state["user"] = user
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid credentials")


def require_auth() -> dict:
    if "user" not in st.session_state:
        login_view()
        st.stop()
    return st.session_state["user"]
