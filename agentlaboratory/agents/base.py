from agentlaboratory.inference import query_model
from agentlaboratory.utils import extract_prompt


class BaseAgent:
    """Base class for implementing research agents.

    This class provides the foundation for creating research agents that can perform
    various research tasks through multiple phases. It handles model interactions,
    maintains conversation history, and manages research state.

    Attributes:
        notes (list): List of notes related to the research task.
        max_steps (int): Maximum number of steps allowed in the research process.
        model (str): Name of the language model to use.
        phases (list): List of research phases.
        plan (str): Research plan.
        report (str): Final research report.
        history (list): Conversation history.
        openai_api_key (str): API key for OpenAI services.
    """

    def __init__(
        self, model="gpt-4o-mini", notes=None, max_steps=100, openai_api_key=None
    ):
        """Initialize the BaseAgent.

        Args:
            model (str, optional): Model identifier. Defaults to "gpt-4o-mini".
            notes (list, optional): Initial notes for the agent. Defaults to None.
            max_steps (int, optional): Maximum steps for research. Defaults to 100.
            openai_api_key (str, optional): OpenAI API key. Defaults to None.
        """
        if notes is None:
            self.notes = []
        else:
            self.notes = notes
        self.max_steps = max_steps
        self.model = model
        self.phases = []
        self.plan = str()
        self.report = str()
        self.history = list()
        self.prev_comm = str()
        self.prev_report = str()
        self.exp_results = str()
        self.dataset_code = str()
        self.results_code = str()
        self.lit_review_sum = str()
        self.interpretation = str()
        self.prev_exp_results = str()
        self.reviewer_response = str()
        self.prev_results_code = str()
        self.prev_interpretation = str()
        self.openai_api_key = openai_api_key

        self.second_round = False
        self.max_hist_len = 15

    def set_model_backbone(self, model):
        """Set the language model to be used by the agent.

        Args:
            model (str): Identifier of the language model.
        """
        self.model = model

    @staticmethod
    def clean_text(text):
        """Clean the text by removing unnecessary newlines in code blocks.

        Args:
            text (str): Text to be cleaned.

        Returns:
            str: Cleaned text.
        """
        text = text.replace("```\n", "```")
        return text

    def inference(self, research_topic, phase, step, feedback="", temp=None):
        """Perform inference using the language model.

        Args:
            research_topic (str): The research topic to investigate.
            phase (str): Current phase of research.
            step (int): Current step number.
            feedback (str, optional): Feedback from previous step. Defaults to "".
            temp (float, optional): Temperature for model sampling. Defaults to None.

        Returns:
            str: Model's response.
        """
        sys_prompt = f"""You are {self.role_description()} \nTask instructions: {self.phase_prompt(phase)}\n{self.command_descriptions(phase)}"""
        context = self.context(phase)
        history_str = "\n".join([_[1] for _ in self.history])
        phase_notes = [_note for _note in self.notes if phase in _note["phases"]]
        notes_str = (
            f"Notes for the task objective: {phase_notes}\n"
            if len(phase_notes) > 0
            else ""
        )
        complete_str = str()
        if step / (self.max_steps - 1) > 0.7:
            complete_str = "You must finish this task and submit as soon as possible!"
        prompt = (
            f"""{context}\n{"~" * 10}\nHistory: {history_str}\n{"~" * 10}\n"""
            f"Current Step #{step}, Phase: {phase}\n{complete_str}\n"
            f"[Objective] Your goal is to perform research on the following topic: {research_topic}\n"
            f"Feedback: {feedback}\nNotes: {notes_str}\nYour previous command was: {self.prev_comm}. Make sure your new output is very different.\nPlease produce a single command below:\n"
        )
        model_resp = query_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=prompt,
            temp=temp,
            openai_api_key=self.openai_api_key,
        )
        print("^" * 50, phase, "^" * 50)
        model_resp = self.clean_text(model_resp)
        self.prev_comm = model_resp
        steps_exp = None
        if feedback is not None and "```EXPIRATION" in feedback:
            steps_exp = int(feedback.split("\n")[0].replace("```EXPIRATION ", ""))
            feedback = extract_prompt(feedback, "EXPIRATION")
        self.history.append(
            (
                steps_exp,
                f"Step #{step}, Phase: {phase}, Feedback: {feedback}, Your response: {model_resp}",
            )
        )
        # remove histories that have expiration dates
        for _i in reversed(range(len(self.history))):
            if self.history[_i][0] is not None:
                self.history[_i] = self.history[_i] = (
                    self.history[_i][0] - 1,
                    self.history[_i][1],
                )
                if self.history[_i][0] < 0:
                    self.history.pop(_i)
        if len(self.history) >= self.max_hist_len:
            self.history.pop(0)
        return model_resp

    def reset(self):
        """Reset the agent's history and previous command."""
        self.history.clear()
        self.prev_comm = ""

    def context(self, phase):
        """Get context information for the current phase.

        Args:
            phase (str): Current phase of research.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def phase_prompt(self, phase):
        """Get the prompt template for the current phase.

        Args:
            phase (str): Current phase of research.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def role_description(self):
        """Get the agent's role description.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def command_descriptions(self, phase):
        """Get available command descriptions for the current phase.

        Args:
            phase (str): Current phase of research.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def example_command(self, phase):
        """Get example commands for the current phase.

        Args:
            phase (str): Current phase of research.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
