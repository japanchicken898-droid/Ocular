import os
import pytest


@pytest.fixture(autouse=True)
def setup_test_auth_env(monkeypatch):
    """Default to allowing unauthenticated requests in existing tests unless auth is explicitly tested."""
    if "OCULAR_API_KEY" not in os.environ and "OCULAR_INSECURE_NO_AUTH" not in os.environ:
        monkeypatch.setenv("OCULAR_INSECURE_NO_AUTH", "1")

