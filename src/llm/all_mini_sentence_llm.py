# -------- knn_feature_extractor_llm.py --------
import os
import json
import pickle
import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from typing import List

from src.helper import functions
from src.llm.sentence_llm_interface import SentenceLLMInterface
from src.models.models import Chunk
from src.templates.template_interface import Template
import src.config as config


class AllMiniSentenceLLM(SentenceLLMInterface):

    def __init__(self,
                 llm_name: str,
                 ):
        super().__init__(llm_name)

    def call(self, msgs: List[Template] | List[str] | List[Chunk]) -> np.array:
        """
        Returns embeddings for a list of Template or string messages.
        """
        input_arr = []
        for msg in msgs:
            msg_str = str(msg)
            input_arr.append(msg_str)

        url = (config.NGROK_URL +
               r'/'
               r'embedd')

        headers = {"Authorization": "Bearer 123"}
        payload = {
            "texts": input_arr,
        }

        response = functions.endpoint_call(
            url=url,
            headers=headers,
            payload=payload
        )

        embeddings = response['response']
        return np.array(embeddings)

    def call_chunks(self, chunks: List[Chunk], batch_size: int) -> List[Chunk]:
        start = 0

        while start < len(chunks):
            end = min(len(chunks), start + batch_size)

            msgs = [str(chunk) for chunk in chunks[start:end]]
            embeddings = self.call(msgs)

            for offset, i in enumerate(range(start, end)):
                chunks[i].embedding = embeddings[offset]

            start = end

        return chunks

    def get_n_dim(self) -> int:
        """
        Return the dimensionality of the embedding vector produced by the feature extractor
        """
        return 384
