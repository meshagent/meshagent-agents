from meshagent.agents.single_shot_writer import SingleShotWriter
from meshagent.openai.tools.responses_adapter import OpenAIResponsesAdapter


class _FakeProtocol:
    def __init__(self, *, token: str | None = None):
        self.token = token


class _FakeRoom:
    def __init__(self, *, token: str | None = None):
        self.protocol = _FakeProtocol(token=token)


def test_single_shot_writer_bind_runtime_credentials_swaps_llm_adapter() -> None:
    adapter = OpenAIResponsesAdapter(model="gpt-4o")
    writer = SingleShotWriter(name="writer", llm_adapter=adapter)

    writer.bind_runtime_credentials(room=_FakeRoom(token="service-token"))

    assert writer._llm_adapter is not adapter
    assert writer._llm_adapter._api_key == "service-token"
