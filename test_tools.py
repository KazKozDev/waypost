import asyncio
from waypost.schemas import ChatRequest, ChatMessage
from waypost.prefix import build_payload
from waypost.registry import Offering

req = ChatRequest(
    messages=[ChatMessage(role="user", content="What is the weather?")],
    tools=[{"type": "function", "function": {"name": "get_weather", "description": "Get weather"}}]
)

o = Offering(
    provider="mlx",
    model_id="mlx-community/Qwen3.6-27B-4bit",
    base_url="http://127.0.0.1:8081/v1",
    tier="L",
    free=True,
    is_local=True,
    caps=["tools", "json", "stream"]
)

payload = build_payload(req, o)
print(payload)
