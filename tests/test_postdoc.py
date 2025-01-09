import pytest
from agentlaboratory.agents.postdoc import PostdocAgent

@pytest.fixture
def mock_api_key():
    return "test-api-key"

@pytest.fixture
def postdoc_agent(mock_api_key):
    return PostdocAgent(openai_api_key=mock_api_key)

def test_postdoc_init(postdoc_agent):
    assert postdoc_agent.phases == ["plan formulation", "results interpretation"]
    assert postdoc_agent.role_description() == "a computer science postdoctoral student at a top university."

def test_postdoc_context_plan_formulation(postdoc_agent):
    postdoc_agent.lit_review_sum = "Test review"
    context = postdoc_agent.context("plan formulation")
    assert isinstance(context, tuple)
    assert "Test review" in context[1]

def test_postdoc_context_invalid_phase(postdoc_agent):
    assert postdoc_agent.context("invalid_phase") is None

def test_command_descriptions(postdoc_agent):
    plan_desc = postdoc_agent.command_descriptions("plan formulation")
    assert "DIALOGUE" in plan_desc
    assert "PLAN" in plan_desc
    
    interp_desc = postdoc_agent.command_descriptions("results interpretation")
    assert "INTERPRETATION" in interp_desc

def test_invalid_phase(postdoc_agent):
    with pytest.raises(Exception, match="Invalid phase"):
        postdoc_agent.command_descriptions("invalid_phase")
