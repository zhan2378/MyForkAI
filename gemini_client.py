# gemini_client.py
from typing import List, Dict
from google import genai


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
        self.client = genai.Client()
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