# Agent Laboratory Documentation

## Overview
The Agent Laboratory implements a collaborative multi-agent system that simulates an academic research environment. The agents work together to conduct research, run experiments, and produce academic papers.

## Agent Hierarchy

BaseAgent ├── ProfessorAgent ├── PostdocAgent
├── MLEngineerAgent └── PhDStudentAgent

## Agent Roles & Responsibilities

### PhDStudentAgent
The central agent that coordinates the research process through multiple phases:
1. Literature Review - Searches and analyzes relevant papers
2. Plan Formulation - Works with PostdocAgent to develop research plan
3. Data Preparation - Directs MLEngineerAgent on data processing
4. Running Experiments - Oversees experiment execution
5. Results Interpretation - Works with PostdocAgent to analyze results
6. Report Writing - Works with ProfessorAgent on paper writing
7. Report Refinement - Handles reviewer feedback

### ProfessorAgent
- Primary role: Guide report writing and provide academic oversight
- Interacts with: PhDStudentAgent
- Key phase: Report Writing
- Capabilities: Generate README.md files, provide paper writing guidance

### PostdocAgent 
- Primary role: Research direction and results analysis
- Interacts with: PhDStudentAgent
- Key phases: Plan Formulation, Results Interpretation
- Capabilities: Review research plans, analyze experimental results

### MLEngineerAgent
- Primary role: Implementation of experiments
- Interacts with: PhDStudentAgent
- Key phases: Data Preparation, Running Experiments
- Capabilities: Code implementation, HuggingFace dataset integration

## Research Workflow

1. Literature Review Phase (PhD Student)

- Searches academic papers via arXiv
- Reviews and summarizes relevant literature
- Builds foundation for research

2. Plan Formulation Phase (PhD Student ↔ Postdoc)

- Discusses research direction
- Develops experimental plan
- Sets methodology and objectives

3. Data Preparation Phase (PhD Student ↔ ML Engineer)

- Searches HuggingFace datasets
- Implements data processing code
- Prepares training/testing data

4. Results Interpretation Phase (PhD Student ↔ Postdoc)

- Analyzes experimental results
- Discusses implications
- Forms conclusions

5. Report Writing Phase (PhD Student ↔ Professor)

- Drafts research paper
- Receives writing feedback
- Produces final LaTeX document

## Communication Protocol

Agents communicate through structured commands:

1. **DIALOGUE**: General communication between agents

dialogue content here

2. Specific Commands by Phase:

- Literature Review:
    - SUMMARY: Search for papers
    - FULL_TEXT: Get full paper text
    - ADD_PAPER: Add paper to review
- Plan Formulation:
    - PLAN: Submit research plan
- Data Preparation:
    - SEARCH_HF: Search HuggingFace datasets
    - SUBMIT_CODE: Submit final code
- Results Interpretation:
    - INTERPRETATION: Submit result analysis
- Report Writing:
    - LATEX: Submit final paper

## Workflow Example

1. Research Initiation:

```python
phd_agent = PhDStudentAgent()
postdoc = PostdocAgent()
ml_engineer = MLEngineerAgent()
professor = ProfessorAgent()

# Start with literature review
research_topic = "Transformer architectures for time series forecasting"
phd_agent.inference(research_topic, phase="literature review", step=0)
```

2. Each agent maintains:

- History of interactions
- Current phase state
- Previous commands
- Research artifacts (code, results, interpretations)

## Customization Options

### 1. Model Selection

You can customize the LLM backend and parameters for each agent:

```python
# Change model backend for all agents
phd_agent = PhDStudentAgent(model="gpt-4")
postdoc = PostdocAgent(model="claude-2")
ml_engineer = MLEngineerAgent(model="gpt-3.5-turbo")

# Adjust temperature during inference
response = phd_agent.inference(
    research_topic="Transformer architectures", 
    phase="literature review",
    step=0,
    temp=0.7  # Higher temperature for more creative responses
)

# Custom API keys
professor = ProfessorAgent(
    model="gpt-4",
    openai_api_key="your-key-here"
)
```

### 2. Research Flow

You can customize the research workflow by modifying agent phases and adding notes:

```python
# Add custom phases to an agent
class CustomPhDAgent(PhDStudentAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phases.append("peer_review")
    
    def phase_prompt(self, phase):
        if phase == "peer_review":
            return "Review other research papers in your field..."
        return super().phase_prompt(phase)

# Add research notes to guide agents
research_notes = [
    {
        "phases": ["literature review", "plan formulation"],
        "content": "Focus on attention mechanisms in time series"
    },
    {
        "phases": ["data preparation"],
        "content": "Ensure data is normalized and stationary"
    }
]

phd_agent = PhDStudentAgent(notes=research_notes)
```

### 3. Context Management

You can customize how agents manage their context and history:

```python
# Adjust history length
phd_agent = PhDStudentAgent(max_steps=200)  # Longer history

# Custom context handling
class ContextAwareAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_context = {}
    
    def context(self, phase):
        base_context = super().context(phase)
        if phase in self.custom_context:
            return base_context + f"\nAdditional context: {self.custom_context[phase]}"
        return base_context

# Reset agent state
phd_agent.reset()  # Clears history and previous commands

# Add custom feedback loop
feedback = {
    "performance_metrics": {"accuracy": 0.95},
    "suggestions": ["Consider broader dataset"]
}
response = phd_agent.inference(
    research_topic="Transformer architectures",
    phase="results interpretation",
    step=5,
    feedback=str(feedback)
)
```

These customizations allow you to:
- Use different language models for different agents
- Add new research phases or modify existing ones
- Provide additional context and guidance through notes
- Adjust how agents maintain state and history
- Implement custom feedback mechanisms

## Key Features

1. State Management

Each agent maintains its state through attributes
History tracking with expiration mechanism
Phase-specific context management

2. Modular Design

BaseAgent provides core functionality
Each specialized agent implements phase-specific behaviors
Clear separation of responsibilities

3. Safety Features

Phase validation
Command validation
History length limits
Error handling for reviews and commands

## Best Practices

1. Initialize agents with appropriate model parameters
2. Follow the phase sequence
3. Handle feedback between phases
4. Monitor agent interactions through history
5. Use notes to provide additional context to agents


## Error Handling

- Phase validation prevents invalid phase transitions
- Command validation ensures proper formatting
- Exception handling for paper reviews and command execution

## Limitations

Requires careful prompt engineering
Dependent on model quality
Sequential phase execution only
Limited memory due to history constraints
