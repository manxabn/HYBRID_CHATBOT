# embeddings.py

import torch
from sentence_transformers import SentenceTransformer

class ChromaEmbeddingFunction:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the SentenceTransformer model. Explicit device selection
        (rather than relying on sentence-transformers' implicit auto-detect)
        so GPU usage is guaranteed, not assumed -- this model is tiny
        (~90MB) and easily fits in VRAM alongside Ollama's resident model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of documents.
        """
        return self.model.encode(texts).tolist()

    def embed_query(self, query_text: str) -> list[float]:
        """
        Generates an embedding for a single query.
        """
        return self.model.encode([query_text])[0].tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        """
        Allows the class instance to be used as a callable for embedding.
        """
        return self.embed_documents(input)
