"""Authentication and access control module for FraudShield MLOps dashboard."""

from typing import Any, Dict, Optional
import streamlit as st

DEMO_USERS = {
    "admin": {
        "password": "admin123",
        "name": "Sarah Connor",
        "role": "Fraud Operations Director",
        "badge": "🛡️ ADMIN",
        "dept": "Risk Governance",
        "avatar": "🛡️",
    },
    "analyst": {
        "password": "analyst123",
        "name": "David Miller",
        "role": "Senior Fraud Analyst",
        "badge": "🔍 ANALYST",
        "dept": "Real-Time Surveillance",
        "avatar": "🔍",
    },
    "auditor": {
        "password": "auditor123",
        "name": "Elena Rostova",
        "role": "Compliance Officer",
        "badge": "⚖️ AUDITOR",
        "dept": "Regulatory Audit",
        "avatar": "⚖️",
    },
}


def verify_credentials(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verifies username and password against registered user credentials."""
    user = DEMO_USERS.get(username.strip().lower())
    if user and user["password"] == password.strip():
        return {
            "username": username.strip().lower(),
            "name": user["name"],
            "role": user["role"],
            "badge": user["badge"],
            "dept": user["dept"],
            "avatar": user["avatar"],
        }
    return None


def init_auth_state():
    """Initializes session state keys for authentication."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user_profile" not in st.session_state:
        st.session_state["user_profile"] = None


def login_user(user_profile: Dict[str, Any]):
    """Marks the user as authenticated in the session state."""
    st.session_state["authenticated"] = True
    st.session_state["user_profile"] = user_profile


def logout_user():
    """Logs the user out and clears session authentication."""
    st.session_state["authenticated"] = False
    st.session_state["user_profile"] = None


def render_login_page():
    """Renders the login view for unauthenticated users."""
    init_auth_state()

    st.markdown(
        """
        <style>
        .login-card {
            background: linear-gradient(145deg, #0f172a, #1e293b);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 2.5rem;
            max-width: 480px;
            margin: 2rem auto;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
            text-align: center;
        }
        .login-title {
            font-size: 1.8rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.25rem;
        }
        .login-subtitle {
            font-size: 0.95rem;
            color: #94a3b8;
            margin-bottom: 2rem;
        }
        .demo-chip {
            background: #1e293b;
            border: 1px solid #475569;
            padding: 0.5rem 0.75rem;
            border-radius: 8px;
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.8, 1])

    with center_col:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <h1 style="font-size: 2.2rem; margin-bottom: 0.2rem;">🛡️ FraudShield MLOps</h1>
                <p style="color: #94a3b8; font-size: 1.05rem;">Production Risk Operations & Surveillance Portal</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("### 🔐 Operator Sign In")
            st.caption("Enter your credentials to access live scoring, cardholder ledgers, and drift monitoring.")

            st.markdown("##### ⚡ Quick One-Click Demo Sign In")
            q_col1, q_col2, q_col3 = st.columns(3)
            if q_col1.button("🛡️ Admin", key="btn_quick_admin", use_container_width=True):
                profile = verify_credentials("admin", "admin123")
                if profile:
                    login_user(profile)
                    st.rerun()

            if q_col2.button("🔍 Analyst", key="btn_quick_analyst", use_container_width=True):
                profile = verify_credentials("analyst", "analyst123")
                if profile:
                    login_user(profile)
                    st.rerun()

            if q_col3.button("⚖️ Auditor", key="btn_quick_auditor", use_container_width=True):
                profile = verify_credentials("auditor", "auditor123")
                if profile:
                    login_user(profile)
                    st.rerun()

            st.divider()

            with st.form("manual_login_form"):
                username_input = st.text_input(
                    "Username",
                    placeholder="e.g. admin or analyst",
                    key="login_user_input",
                )
                password_input = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter password",
                    key="login_pass_input",
                )
                submit_login = st.form_submit_button(
                    "Sign In",
                    type="primary",
                    use_container_width=True,
                )

                if submit_login:
                    profile = verify_credentials(username_input, password_input)
                    if profile:
                        login_user(profile)
                        st.success(f"Welcome back, {profile['name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password. Use demo accounts or check credentials.")

            st.caption(
                "💡 **Demo Logins:** `admin` / `admin123` &bull; `analyst` / `analyst123` &bull; `auditor` / `auditor123`"
            )

        st.caption(
            "🔒 **Security Notice:** Restricted financial infrastructure. All transactions, drift checks, and profile modifications are cryptographically signed and logged."
        )
