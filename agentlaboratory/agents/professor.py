from .base import BaseAgent
from utils import query_model

class ProfessorAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = ["report writing"]

    def generate_readme(self):
        sys_prompt = f"""You are {self.role_description()} \n Here is the written paper \n{self.report}. Task instructions: Your goal is to integrate all of the knowledge, code, reports, and notes provided to you and generate a readme.md for a github repository."""
        history_str = "\n".join([_[1] for _ in self.history])
        prompt = (
            f"""History: {history_str}\n{'~' * 10}\n"""
            f"Please produce the readme below in markdown:\n")
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt, openai_api_key=self.openai_api_key)
        return model_resp.replace("```markdown", "")

    def context(self, phase):
        return ""

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return (
            "When you believe a good report has been arrived at between you and the PhD student you can use the following command to end the dialogue and submit the plan ```LATEX\nreport here\n```\n where report here is the actual report written in compilable latex to be transmitted and LATEX is just the word LATEX.\n"
            "Your report should include numbers, relevant metrics to the experiment (e.g. accuracy or loss) and measures of significance. You must propagate this information accurately. You must also submit the report promptly. Do not delay too long.\n"
            "You must be incredibly detailed about what you did for the experiment and all of the findings.\n"
            )

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        phase_str = (
            "You are directing a PhD student to help them write a report in latex based on results from an experiment, and you interact with them through dialogue.\n"
            "Your goal is to write a report in latex for an experiment. You should read through the code, read through the interpretation, and look at the results to understand what occurred. You should then discuss with the PhD student how they can write up the results and give their feedback to improve their thoughts.\n"
        )
        return phase_str

    def role_description(self):
        return "a computer science professor at a top university."

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return (
            "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
            "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\n<Insert command here>\n``` where COMMAND is the specific command you want to run (e.g. REPORT, DIALOGUE).\n")
