from .base import BaseAgent


class MLEngineerAgent(BaseAgent):
    def __init__(
        self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None
    ):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = [
            "data preparation",
            "running experiments",
        ]

    def context(self, phase):
        sr_str = str()
        if self.second_round:
            sr_str = (
                f"The following are results from the previous experiments\n",
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n",
            )
        if phase == "data preparation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\nPlan: {self.plan}",
                f"Current Plan: {self.plan}",
            )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            return (
                "You can produce code using the following command: ```python\ncode here\n```\n where code here is the actual code you will execute in a Python terminal, and python is just the word python. Try to incorporate some print functions. Do not use any classes or functions. If your code returns any errors, they will be provided to you, and you are also able to see print statements. You will receive all print statement results from the code. Make sure function variables are created inside the function or passed as a function parameter.\n"
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send, and DIALOGUE is just the word DIALOGUE.\n"
                "You also have access to HuggingFace datasets. You can search the datasets repository using the following command: ```SEARCH_HF\nsearch query here\n``` where search query here is the query used to search HuggingFace datasets, and SEARCH_HF is the word SEARCH_HF. This will return a list of HuggingFace dataset descriptions which can be loaded into Python using the datasets library. Your code MUST use an external HuggingFace directory.\n"
                "You MUST use a HuggingFace dataset in your code. DO NOT CREATE A MAIN FUNCTION. Try to make the code very simple.\n"
                "You can only use a SINGLE command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, NOT BOTH.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. python, DIALOGUE, SEARCH_HF).\n"
            )
        return ()

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            phase_str = (
                "You are a machine learning engineer being directed by a PhD student who will help you write the code, and you can interact with them through dialogue.\n"
                "Your goal is to produce code that prepares the data for the provided experiment. You should aim for simple code to prepare the data, not complex code. You should integrate the provided literature review and the plan and come up with code to prepare data for this experiment.\n"
            )
        return phase_str

    def role_description(self):
        return "a machine learning engineer working at a top university."
