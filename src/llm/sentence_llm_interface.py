from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.templates.template_interface import *


class SentenceLLMInterface(ABC):
    def __init__(self, llm_name: str):
        self._name = llm_name

    @abstractmethod
    def call(self, msgs: List[Template] | List[str]) -> np.array:
        pass

    @abstractmethod
    def call_chunks(self, chunks: List[Chunk], batch_size: int) -> List[Chunk]:
        pass

    @abstractmethod
    def get_n_dim(self) -> int:
        pass
