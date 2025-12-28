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

from src.llm.sentence_llm_interface import SentenceLLMInterface
from src.models.models import Chunk
from src.templates.template_interface import Template


class KNNFeatureExtractorLLM(SentenceLLMInterface):

    def __init__(self,
                 llm_name: str,
                 ):
        super().__init__(llm_name)
        self.max_len = 12

        # -----------------------------
        # Load Tokenizer
        # -----------------------------
        tokenizer_path = "assets/knn_llm/tokenizer.pkl"
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        # -----------------------------
        # Load Feature Extractor Model
        # -----------------------------
        feature_extractor_path = "assets/knn_llm/lstm_shallow_attention_v2_feature_extractor.keras"
        self.feature_extractor = load_model(feature_extractor_path)

    def _process_pattern(self, text: str) -> str:
        """
        Preprocess input text for tokenization.
        """
        text = str(text).lower().strip()
        text = re.sub(r"[^A-Za-z0-9]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
        text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
        return text

    def call(self, msgs: List[Template] | List[str] | List[Chunk]) -> np.array:
        """
        Returns embeddings for a list of Template or string messages.
        """
        input_arr = []
        for msg in msgs:
            msg_str = str(msg)
            msg_str = self._process_pattern(msg_str)
            seq = self.tokenizer.texts_to_sequences([msg_str])
            seq = pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")
            input_arr.append(seq[0])

        input_arr = np.array(input_arr)

        all_embeddings = self.feature_extractor.predict(input_arr, verbose=0)
        return np.array(all_embeddings)

    def call_chunks(self, chunks: List[Chunk], batch_size: int) -> List[Chunk]:
        for i in range(len(chunks)):
            msg = chunks[i].txt
            embedding = self.call([msg])[0]
            chunks[i].embedding = embedding
        return chunks

    def get_n_dim(self) -> int:
        """
        Return the dimensionality of the embedding vector produced by the feature extractor
        """
        return self.feature_extractor.output_shape[-1]
