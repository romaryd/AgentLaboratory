import pytest
from agentlaboratory.agents.mlengineer import MLEngineerAgent

def test_initialization():
    agent = MLEngineerAgent()
    assert agent.model == "gpt4omini"
    assert agent.max_steps == 100
    assert agent.phases == ["data preparation", "running experiments"]

def test_context():
    agent = MLEngineerAgent()
    agent.lit_review_sum = "Test literature review"
    agent.plan = "Test plan"
    
    # Test data preparation phase
    context = agent.context("data preparation")
    assert isinstance(context, tuple)
    assert "Test literature review" in context[1]
    assert "Test plan" in context[1]
    
    # Test invalid phase
    assert agent.context("invalid_phase") == ""

def test_example_command():
    agent = MLEngineerAgent()
    
    # Test valid phase
    assert agent.example_command("data preparation") == ()
    
    # Test invalid phase
    with pytest.raises(Exception) as exc_info:
        agent.example_command("invalid_phase")
    assert "Invalid phase" in str(exc_info.value)

def test_command_descriptions():
    agent = MLEngineerAgent()
    
    # Test data preparation phase
    desc = agent.command_descriptions("data preparation")
    assert isinstance(desc, str)
    assert "DIALOGUE" in desc
    assert "SEARCH_HF" in desc
    assert "python" in desc
    
    # Test invalid phase
    with pytest.raises(Exception) as exc_info:
        agent.command_descriptions("invalid_phase")
    assert "Invalid phase" in str(exc_info.value)

def test_phase_prompt():
    agent = MLEngineerAgent()
    
    # Test data preparation phase
    prompt = agent.phase_prompt("data preparation")
    assert isinstance(prompt, str)
    assert "machine learning engineer" in prompt
    
    # Test invalid phase
    with pytest.raises(Exception) as exc_info:
        agent.phase_prompt("invalid_phase")
    assert "Invalid phase" in str(exc_info.value)

def test_role_description():
    agent = MLEngineerAgent()
    desc = agent.role_description()
    assert isinstance(desc, str)
    assert "machine learning engineer" in desc
    assert "university" in desc
