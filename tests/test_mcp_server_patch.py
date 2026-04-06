import asyncio

from pulsar.mcp import server


class _DummyTool:
    parameters = {"properties": {"dataset_geometry": {"type": "string"}}}


def test_lenient_function_tool_run_strips_unknown_fields(monkeypatch):
    captured = {}

    async def fake_run(self, arguments):
        captured["self"] = self
        captured["arguments"] = arguments
        return "ok"

    monkeypatch.setattr(server, "_original_function_tool_run", fake_run)

    result = asyncio.run(
        server._lenient_function_tool_run(
            _DummyTool(),
            {
                "dataset_geometry": "dense numeric manifold",
                "wait_for_previous": True,
            },
        )
    )

    assert result == "ok"
    assert captured["arguments"] == {"dataset_geometry": "dense numeric manifold"}


def test_lenient_function_tool_run_preserves_non_dict_arguments(monkeypatch):
    captured = {}

    async def fake_run(self, arguments):
        captured["arguments"] = arguments
        return "ok"

    monkeypatch.setattr(server, "_original_function_tool_run", fake_run)

    result = asyncio.run(server._lenient_function_tool_run(_DummyTool(), None))

    assert result == "ok"
    assert captured["arguments"] is None
