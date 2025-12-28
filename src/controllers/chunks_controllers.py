from typing import List
import numpy as np

from src.models.models import Chunk


class ChunksController:
    def __init__(self, window_size: int, overlap_size: int):
        """
        window_size   -> number of words per chunk
        overlap_size  -> number of overlapping words between chunks
        """
        self.window_size = window_size
        self.overlap_size = overlap_size

    def no_slice(self, chunks: List[str]) -> List[Chunk]:
        return list(map(lambda x: Chunk(
            txt=x,
            embedding=np.zeros(1),  # Placeholder for embedding
            metadata={}
        ), chunks))

    def fixed_window_slicer(self, msg: str) -> List[Chunk]:
        """
        Splits the input message into word-based chunks
        with overlap (also in words) and removes duplicate chunks.
        """
        chunks: List[Chunk] = []
        seen_texts = set()

        words = str(msg).strip().split()
        total_words = len(words)
        start = 0

        while start < total_words:
            end = min(start + self.window_size, total_words)
            text_chunk = " ".join(words[start:end])

            # Skip duplicate chunks
            if text_chunk not in seen_texts:
                chunk = Chunk(
                    txt=text_chunk,
                    embedding=np.zeros(1),  # Placeholder for embedding
                    metadata={
                        "start_word": start,
                        "end_word": end,
                        "word_count": end - start
                    }
                )
                chunks.append(chunk)
                seen_texts.add(text_chunk)

            # Move window forward with overlap
            start += self.window_size - self.overlap_size

            # Safety check (avoid infinite loop)
            if self.window_size <= self.overlap_size:
                break

        return chunks
