import json
import unittest

from agents_v2.reasoner import Reasoner, ReasonerLLMConfig
from agents_v2.schemas import Observation


class ReasonerLLMStubTests(unittest.TestCase):
    def _make_observation(self, node_id: str, text: str) -> Observation:
        return Observation(node_id=node_id, modality="text", payload={"text": text})

    def test_llm_reasoner_returns_parsed_answer(self) -> None:
        call_log = {}

        def stub_llm(**kwargs):
            call_log["called"] = True
            evidence = json.loads(kwargs["context"])
            self.assertTrue(any(block["node_id"] == "sec_1" for block in evidence))
            return json.dumps(
                {
                    "answer": "42% of land area",
                    "confidence": 0.9,
                    "support_nodes": ["sec_1"],
                    "reasoning": "LLM analysed evidence",
                }
            )

        reasoner = Reasoner(use_llm=True, llm_callable=stub_llm, llm_config=ReasonerLLMConfig())
        snapshot = {
            "observations": {
                "sec_1": self._make_observation("sec_1", "Bronx land area rezoned 42%"),
            }
        }

        answer = reasoner.run("What percentage?", snapshot)
        self.assertTrue(call_log.get("called"))
        self.assertEqual(answer.answer, "42% of land area")
        self.assertEqual(answer.support_nodes, ["sec_1"])
        self.assertGreaterEqual(answer.confidence, 0.9)
        self.assertIn("LLM analysed evidence", answer.reasoning_trace[0])

    def test_reasoner_replans_when_llm_fails(self) -> None:
        def failing_llm(**kwargs):
            return ""

        reasoner = Reasoner(use_llm=True, llm_callable=failing_llm)
        snapshot = {
            "observations": {
                "sec_1": self._make_observation("sec_1", "Bronx land area rezoned 42%"),
            }
        }

        answer = reasoner.run("What percentage?", snapshot)
        self.assertEqual(answer.action, "REPLAN")
        self.assertEqual(answer.missing_intent, {"need": "llm_response"})

    def test_reasoner_includes_table_structure_in_prompt(self) -> None:
        captured = {}

        def stub_llm(**kwargs):
            payload = json.loads(kwargs["context"])
            captured["context"] = payload
            return json.dumps(
                {
                    "answer": "1.5%",
                    "confidence": 0.8,
                    "support_nodes": ["tab_1"],
                }
            )

        reasoner = Reasoner(use_llm=True, llm_callable=stub_llm)
        table_payload = {
            "text": "Table listing percentages",
            "structured_table": {
                "columns": ["Borough", "% Land Area"],
                "rows": [["Bronx", "1.5%"], ["Queens", "2.0%"]],
                "caption": "Rezoned land area",
            },
        }
        snapshot = {
            "observations": {
                "tab_1": Observation(node_id="tab_1", modality="table", payload=table_payload)
            }
        }

        answer = reasoner.run("Bronx land area?", snapshot)
        ctx = captured["context"]
        self.assertTrue(any(block.get("table") for block in ctx))
        self.assertEqual(answer.answer, "1.5%")


if __name__ == "__main__":
    unittest.main()
