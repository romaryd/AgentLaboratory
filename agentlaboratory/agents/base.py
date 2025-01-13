from agentlaboratory.inference import query_model
from agentlaboratory.utils import extract_prompt


class BaseAgent:
    def __init__(
        self, model="gpt-4o-mini", notes=None, max_steps=100, openai_api_key=None
    ):
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
        self.model = model

    @staticmethod
    def clean_text(text):
        text = text.replace("```\n", "```")
        return text

    def inference(self, research_topic, phase, step, feedback="", temp=None):
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
        self.history.clear()
        self.prev_comm = ""

    def context(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")

    def phase_prompt(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")

    def role_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def command_descriptions(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")

    def example_command(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")
