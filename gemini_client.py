# gemini_client.py
from typing import List, Dict
from google import genai
import os


class GeminiClient:
    """
    Gemini provider wrapper.
    Uses google-genai SDK exactly as in official examples.
    Requires:
      - pip install google-genai
      - GEMINI_API_KEY or GOOGLE_API_KEY in environment
    """

    def __init__(
        self,
        model: str = "gemini-3-pro-preview",
        embed_model: str = "gemini-embedding-001",
    ):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        # 🔑 Explicitly pass api_key (required by your SDK version)
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.embed_model = embed_model

    def respond(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> str:
        """
        Flatten multi-turn messages into a single prompt string.
        """

        system_lines = []
        convo_lines = []

        for m in messages:
            role = m["role"]
            text = m["content"].strip()
            if role == "system":
                system_lines.append(text)
            else:
                convo_lines.append(f"{role.upper()}: {text}")

        prompt = ""
        if system_lines:
            prompt += "SYSTEM:\n" + "\n".join(system_lines) + "\n\n"

        prompt += "\n\n".join(convo_lines)
        prompt += "\n\nASSISTANT:"

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return (response.text or "").strip()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Gemini embedding model.
        """
        contents = texts if len(texts) > 1 else texts[0]

        result = self.client.models.embed_content(
            model=self.embed_model,
            contents=contents,
        )

        return [list(e.values) for e in result.embeddings]