
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json"  # or any specific file you want
)


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def __call__(self, texts):
        return [self.model.encode(t, convert_to_numpy=True).tolist() for t in texts]


"""
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2",device="cpu")

    def __call__(self, input):
        return self.model.encode(input).tolist()
"""