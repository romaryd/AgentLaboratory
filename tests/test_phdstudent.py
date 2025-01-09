import pytest
from unittest.mock import patch, MagicMock
from agentlaboratory.agents.phdstudent import PhDStudentAgent

@pytest.fixture
def mock_api_key():
    return "test-api-key"

@pytest.fixture
def phd_agent(mock_api_key):
    return PhDStudentAgent(openai_api_key=mock_api_key)

def test_phd_init(phd_agent):
    expected_phases = [
        "literature review",
        "plan formulation",
        "data preparation",
        "running experiments",
        "results interpretation",
        "report writing",
        "report refinement",
    ]
    assert phd_agent.phases == expected_phases
    assert phd_agent.lit_review == []
    assert phd_agent.role_description() == "a computer science PhD student at a top university."

def test_phd_context_literature_review(phd_agent):
    context = phd_agent.context("literature review")
    assert context == ""

def test_phd_context_plan_formulation(phd_agent):
    phd_agent.lit_review_sum = "Test review"
    context = phd_agent.context("plan formulation")
    assert isinstance(context, tuple)
    assert "Test review" in context[1]

@patch('agentlaboratory.agents.phdstudent.query_model')
def test_requirements_txt(mock_query, phd_agent):
    mock_query.return_value = "numpy==1.21.0\npandas==1.3.0"
    result = phd_agent.requirements_txt()
    assert "numpy" in result
    mock_query.assert_called_once()

def test_add_review(phd_agent):
    class MockArxivEngine:
        def retrieve_full_paper_text(self, arxiv_id):
            return "Full paper text"

    review = "2301.00001\nThis is a test paper summary"
    message, full_text = phd_agent.add_review(review, MockArxivEngine())
    
    assert "Successfully added" in message
    assert len(phd_agent.lit_review) == 1
    assert phd_agent.lit_review[0]["arxiv_id"] == "2301.00001"
    assert phd_agent.lit_review[0]["summary"] == "This is a test paper summary"

def test_format_review(phd_agent):
    phd_agent.lit_review = [{
        "arxiv_id": "2301.00001",
        "summary": "Test summary",
        "full_text": "Full text"
    }]
    formatted = phd_agent.format_review()
    assert "2301.00001" in formatted
    assert "Test summary" in formatted

def test_invalid_phase(phd_agent):
    with pytest.raises(Exception, match="Invalid phase"):
        phd_agent.command_descriptions("invalid_phase")
