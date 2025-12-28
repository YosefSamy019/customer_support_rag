import faiss
import numpy as np
import os
import pickle
from typing import List

from src.database.vector_db_interface import VectorDBInterface
from src.models.models import Chunk


class FAISSCpuVectorDB(VectorDBInterface):
    def __init__(self, dim_len: int, cache_path: str, top_k: int) -> None:
        """
        dim_len: dimension of embeddings
        cache_path: path to save/load chunk metadata
        """
        super().__init__(dim_len, cache_path)
        self.dim_len = dim_len
        self.top_k = top_k
        self.cache_path = cache_path
        self.chunks: List[Chunk] = []

        # Load only chunks from cache
        self.load_from_cache()

    def _validate_embedding(self, embedding):
        """
        Ensure the embedding is a numpy array of correct shape and type.
        """
        embedding = np.array(embedding, dtype='float32')
        if embedding.shape[0] != self.dim_len:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim_len}, got {embedding.shape[0]}")
        return embedding

    def _build_index(self):
        """
        Build a FAISS index from the current chunks (on the fly).
        """
        self.index = faiss.IndexFlatL2(self.dim_len)
        if self.chunks:
            embeddings = np.array([self._validate_embedding(c.embedding) for c in self.chunks], dtype='float32')
            self.index.add(embeddings)

    def store_chunk(self, chunk: Chunk):
        """
        Store a single chunk in memory and save chunks to cache.
        """
        self.chunks.append(chunk)
        self.save_to_cache()

    def store_chunks(self, chunks: List[Chunk]):
        """
        Store multiple chunks at once in memory and save chunks to cache.
        """
        if not chunks:
            return
        self.chunks.extend(chunks)
        self.save_to_cache()

    def search_embedding(self, embedding) -> List[Chunk]:
        """
        Search for top_k nearest chunks to the given embedding.
        FAISS index is created on the fly from the current chunks.
        """
        if not self.chunks:
            return []

        self._build_index()
        embedding_np = np.array([self._validate_embedding(embedding)], dtype='float32')
        distances, indices = self.index.search(embedding_np, self.top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    def save_to_cache(self):
        """
        Save only the chunk metadata using pickle.
        """
        try:
            with open(self.cache_path + "_chunks.pkl", "wb") as f:
                pickle.dump(self.chunks, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def load_from_cache(self):
        """
        Load chunk metadata from disk if it exists.
        """
        try:
            if os.path.exists(self.cache_path + "_chunks.pkl"):
                with open(self.cache_path + "_chunks.pkl", "rb") as f:
                    self.chunks = pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")

    def get_all_chunks(self) -> List[Chunk]:
        return self.chunks

    def filter_new_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Given a list of chunks, return only those that do not exist in the cached chunks.
        Comparison is based on chunk.txt.
        """
        cached_texts = set(c.txt for c in self.chunks)
        return [c for c in chunks if c.txt not in cached_texts]
