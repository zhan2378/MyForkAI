from __future__ import annotations
from oa_client import OAClient
from gemini_client import GeminiClient

def make_provider(provider: str, model: str, embed_model: str):
    p = provider.lower()
    if p in ("openai", "gpt"):
        return OAClient(model=model, embed_model=embed_model)
    if p in ("gemini", "google"):
        return GeminiClient(model=model, embed_model=embed_model)
    raise ValueError(f"Unknown provider: {provider}")