from typing import List

from src.controllers.chunks_controllers import ChunksController
from src.controllers.ingestion_controllers import IngestionController
from src.database.fiass_vector_db import FAISSCpuVectorDB
from src.database.vector_db_interface import VectorDBInterface
from src.llm.all_mini_sentence_llm import AllMiniSentenceLLM
from src.llm.gemini_llm import GeminiLLM
from src.llm.sentence_llm_interface import SentenceLLMInterface
from src.llm.seq2seq_llm_interface import Seq2SeqLLMInterface
from src.models.models import ChatMsg


class Project:
    def __init__(self,
                 project_name: str,
                 vector_db_path: str,
                 ):
        self.project_name: str = project_name

        self.chat_history: List[ChatMsg] = []

        self.ingestion_controller: IngestionController = IngestionController()

        self.chunks_controller: ChunksController = ChunksController(
            window_size=25,
            overlap_size=2,
        )

        self.QA_llm: Seq2SeqLLMInterface = GeminiLLM("gemini")

        self.intent_rephraser_llm: Seq2SeqLLMInterface = GeminiLLM("gemini")

        # self.sentence_llm: SentenceLLMInterface = KNNFeatureExtractorLLM("knn sentence llm", )
        self.sentence_llm: SentenceLLMInterface = AllMiniSentenceLLM("knn sentence llm", )

        self.vector_db: VectorDBInterface = FAISSCpuVectorDB(
            dim_len=self.sentence_llm.get_n_dim(),
            cache_path=vector_db_path,
            top_k=3,
        )
