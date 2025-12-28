from abc import ABC, abstractmethod
from typing import List

from src.models.models import *


class VectorDBInterface(ABC):
    def __init__(self, dim_len: int, cache_path: str):
        pass

    @abstractmethod
    def store_chunk(self, chunk: Chunk):
        pass

    @abstractmethod
    def store_chunks(self, chunks: List[Chunk]):
        pass

    @abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        pass

    @abstractmethod
    def search_embedding(self, embedding) -> List[Chunk]:
        pass

    @abstractmethod
    def save_to_cache(self):
        pass

    @abstractmethod
    def load_from_cache(self):
        pass

    @abstractmethod
    def filter_new_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        pass
