import numpy as np
from sentence_transformers import SentenceTransformer
from .preprocess import normalize_text

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts):
        texts = [normalize_text(t or "") for t in texts]
        arr = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(arr, dtype=np.float32)

def cosine(a, b):
    return float(np.dot(a, b))
