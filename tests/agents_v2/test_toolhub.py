import unittest

from agents_v2.toolhub import ToolCall, ToolExecutor, ToolRegistry, ToolResult


class ToolHubTests(unittest.TestCase):
    def test_registry_executes_registered_tool(self) -> None:
        registry = ToolRegistry()

        def sample_tool(call: ToolCall) -> ToolResult:
            return ToolResult(
                status="ok",
                data={"echo": call.args},
                metrics={"custom": 1},
            )

        registry.register("echo.tool", sample_tool)
        executor = ToolExecutor(registry)
        result = executor.run(ToolCall(tool_id="echo.tool", args={"msg": "hi"}))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["echo"]["msg"], "hi")
        self.assertIn("latency_ms", result.metrics)
        self.assertEqual(result.metrics["attempt"], 1)

    def test_missing_tool_raises(self) -> None:
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        with self.assertRaises(KeyError):
            executor.run(ToolCall(tool_id="missing.tool", args={}))


if __name__ == "__main__":
    unittest.main()
