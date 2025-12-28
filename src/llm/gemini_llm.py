from src.llm.seq2seq_llm_interface import *
import src.helper.functions as functions
import os
from google import genai


class GeminiLLM(Seq2SeqLLMInterface):
    GOOGLE_API_KEY = "AIzaSyAdhXG91SsmuO30DLjvE35Tsx0USPlW8xg"
    MODEL = "gemini-2.5-flash"

    def __init__(
            self,
            llm_name: str,
    ):
        super().__init__(llm_name)
        self.llm_name = llm_name
        self.client = genai.Client(
            api_key=GeminiLLM.GOOGLE_API_KEY,
        )

    def call(self, msg: Template | str) -> str:
        try:
            # Send the request to generate content
            response = self.client.models.generate_content(
                model=GeminiLLM.MODEL,
                contents=str(msg)
            )
            return response.text
        except Exception as e:
            return f"API request failed: {e}"
