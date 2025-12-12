from openai import OpenAI

class OAClient:
    def __init__(self, model, embed_model):
        self.client = OpenAI()
        self.model = model
        self.embed_model = embed_model

    def respond(self, messages, temperature=0.2, max_tokens=800):
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        r = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return r.output_text

    def embed(self, texts):
        r = self.client.embeddings.create(
            model=self.embed_model,
            input=texts
        )
        return [d.embedding for d in r.data]