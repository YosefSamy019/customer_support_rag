import json
import pandas as pd
from typing import Union, List
from pathlib import Path
import PyPDF2


class IngestionController:
    def __init__(self):
        pass  # Add any initialization if needed

    def ingest(self, msg: Union[str, int, float, list, tuple, dict]) -> str:
        """
        Convert input message to string.
        - If str/int/float, return as string.
        - If list/tuple/dict, convert to JSON string.
        - Raise ValueError otherwise.
        """
        if isinstance(msg, (str, int, float)):
            return str(msg)
        if isinstance(msg, (list, tuple, dict)):
            return json.dumps(msg, ensure_ascii=False)
        raise ValueError("Type not supported. Must be str, list, tuple, dict, int, or float.")

    def ingest_from_pdf(self, pdf_path: str) -> str:
        """
        Reads a PDF file and returns the extracted text as a string.
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text = ""
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"

        return text.strip()

    def ingest_from_json(self, json_path: str) -> List[str]:
        """
        Reads a JSON file containing a list of objects with 'question' and 'answer' keys.
        Returns a list of strings where each string is "question answer".
        """
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON must be a list of objects with 'question' and 'answer' keys.")

        combined = []
        for item in data:
            if not {"question", "answer"}.issubset(item):
                raise ValueError("Each object in JSON must contain 'question' and 'answer' keys.")
            q = str(item["question"]).strip()
            a = str(item["answer"]).strip()
            combined.append(f"{q} {a}")

        return combined
