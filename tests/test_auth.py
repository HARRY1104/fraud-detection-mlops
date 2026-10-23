from src.app.auth import DEMO_USERS, verify_credentials


def test_verify_valid_credentials():
    admin = verify_credentials("admin", "admin123")
    assert admin is not None
    assert admin["role"] == "Fraud Operations Director"
    assert admin["username"] == "admin"

    analyst = verify_credentials("analyst", "analyst123")
    assert analyst is not None
    assert analyst["role"] == "Senior Fraud Analyst"

    auditor = verify_credentials("auditor", "auditor123")
    assert auditor is not None
    assert auditor["role"] == "Compliance Officer"


def test_verify_invalid_credentials():
    # Wrong password
    assert verify_credentials("admin", "wrongpass") is None

    # Unknown username
    assert verify_credentials("nobody", "admin123") is None

    # Empty inputs
    assert verify_credentials("", "") is None


def test_demo_users_structure():
    for uname, user in DEMO_USERS.items():
        assert "password" in user
        assert "name" in user
        assert "role" in user
        assert "badge" in user
