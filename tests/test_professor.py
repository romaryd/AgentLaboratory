import pytest
from unittest.mock import patch, MagicMock
from agentlaboratory.agents.professor import ProfessorAgent

@pytest.fixture
def mock_api_key():
    return "test-api-key"

@pytest.fixture
def professor_agent(mock_api_key):
    return ProfessorAgent(openai_api_key=mock_api_key)

def test_professor_init(professor_agent):
    assert professor_agent.phases == ["report writing"]
    assert professor_agent.role_description() == "a computer science professor at a top university."

def test_professor_context(professor_agent):
    assert professor_agent.context("report writing") == ""

@patch('agentlaboratory.agents.professor.query_model')
def test_generate_readme(mock_query, professor_agent):
    mock_query.return_value = "# Test Project\nTest readme content"
    professor_agent.report = "Test report"
    result = professor_agent.generate_readme()
    assert "Test Project" in result
    mock_query.assert_called_once()

def test_invalid_phase(professor_agent):
    with pytest.raises(Exception, match="Invalid phase"):
        professor_agent.command_descriptions("invalid_phase")
