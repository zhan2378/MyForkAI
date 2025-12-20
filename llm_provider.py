from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Protocol

Message = Dict[str, str]  # {"role": "system|user|assistant", "content": "..."}

class LLMProvider(Protocol):
    def respond(self, messages: List[Message], temperature: float, max_tokens: int) -> str: ...
    def embed(self, texts: List[str]) -> List[list]: ...