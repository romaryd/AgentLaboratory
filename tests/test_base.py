import unittest
from agentlaboratory.agents.base import BaseAgent


class TestBaseAgent(unittest.TestCase):

    def setUp(self):
        self.agent = BaseAgent()

    def test_initialization(self):
        self.assertEqual(self.agent.model, "gpt-4o-mini")
        self.assertEqual(self.agent.notes, [])
        self.assertEqual(self.agent.max_steps, 100)
        self.assertEqual(self.agent.openai_api_key, None)
        self.assertEqual(self.agent.phases, [])
        self.assertEqual(self.agent.plan, "")
        self.assertEqual(self.agent.report, "")
        self.assertEqual(self.agent.history, [])
        self.assertEqual(self.agent.prev_comm, "")
        self.assertEqual(self.agent.prev_report, "")
        self.assertEqual(self.agent.exp_results, "")
        self.assertEqual(self.agent.dataset_code, "")
        self.assertEqual(self.agent.results_code, "")
        self.assertEqual(self.agent.lit_review_sum, "")
        self.assertEqual(self.agent.interpretation, "")
        self.assertEqual(self.agent.prev_exp_results, "")
        self.assertEqual(self.agent.reviewer_response, "")
        self.assertEqual(self.agent.prev_results_code, "")
        self.assertEqual(self.agent.prev_interpretation, "")
        self.assertEqual(self.agent.second_round, False)
        self.assertEqual(self.agent.max_hist_len, 15)

    def test_set_model_backbone(self):
        self.agent.set_model_backbone("new-model")
        self.assertEqual(self.agent.model, "new-model")

    def test_clean_text(self):
        dirty_text = "Some text with a code block\n```\ncode\n```"
        clean_text = self.agent.clean_text(dirty_text)
        self.assertEqual(clean_text, "Some text with a code block\n```code\n```")

    def test_inference(self):
        # Mocking the methods that are not implemented
        self.agent.role_description = lambda: "Research Assistant"
        self.agent.phase_prompt = lambda phase: "Phase prompt"
        self.agent.command_descriptions = lambda phase: "Command descriptions"
        self.agent.context = lambda phase: "Context"

        research_topic = "AI Research"
        phase = "Phase 1"
        step = 1
        feedback = "Good job"
        temp = 0.7

        # Mocking query_model function
        def mock_query_model(model_str, system_prompt, prompt, temp, openai_api_key):
            return "Mocked response"

        self.agent.query_model = mock_query_model

        response = self.agent.inference(research_topic, phase, step, feedback, temp)
        self.assertEqual(response, "Mocked response")
        self.assertEqual(self.agent.prev_comm, "Mocked response")
        self.assertEqual(len(self.agent.history), 1)

    def test_reset(self):
        self.agent.history.append("some history")
        self.agent.prev_comm = "some command"
        self.agent.reset()
        self.assertEqual(self.agent.history, [])
        self.assertEqual(self.agent.prev_comm, "")

if __name__ == '__main__':
    unittest.main()