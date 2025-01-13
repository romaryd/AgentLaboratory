import pytest


@pytest.fixture
def mock_model_name():
    return "gpt4omini"


@pytest.fixture
def mock_api_response():
    return {"choices": [{"text": "Test response"}]}


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
