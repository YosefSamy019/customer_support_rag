from enum import Enum


class Chunk:
    def __init__(self, txt: str, embedding, metadata: dict):
        self.txt = str(txt)
        self.embedding = embedding
        self.metadata = dict(metadata)

    def to_dict(self):
        return {
            "txt": self.txt,
            # "embedding": self.embedding,
            "metadata": self.metadata,
        }

    def __str__(self):
        return self.txt


class ChatSender(Enum):
    user = 0
    assistant = 1


class ChatMsg:
    def __init__(self, msg: str, sender: ChatSender, metadata=None):
        self.msg = str(msg)
        self.sender = sender
        self.metadata = dict(metadata) if metadata is not None else metadata
