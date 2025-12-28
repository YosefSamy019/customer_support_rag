from abc import ABC, abstractmethod
from typing import List

from src.models.models import Chunk, ChatMsg


class Template(ABC):
    @abstractmethod
    def __str__(self):
        pass

    def as_str(self):
        return str(self)


class CompanyTemplate(Template):
    def __init__(self, company_name: str = "Yosef for Trades", phone: str = "+201030296141",
                 email: str = "yosefsamy019@gmail.com"):
        self.company_name = company_name
        self.phone = phone
        self.email = email

    def __str__(self) -> str:
            return f"""
COMPANY INFORMATION (CONDITIONAL):

Only include the following information in your response IF AND ONLY IF:
- The user explicitly asks for company details, contact information, phone number, email, or how to get in touch.
- Do NOT include this information otherwise.

If requested, respond exactly with:

Company Name: {self.company_name}
Phone: {self.phone}
Email: {self.email}

Do NOT rephrase, summarize, or add extra text.
    """.strip()


class QueryChunksAnsweringTemplate(Template):
    def __init__(self, query: str, candidate_chunks: List[Chunk]):
        self.query = query
        self.candidate_chunks = candidate_chunks

    def __str__(self) -> str:
        chunks_text = ",\n".join(f'"{c}"' for c in self.candidate_chunks)

        return f"""System:
You are a retrieval-based QA system.

SPECIAL RULE — GREETINGS:
If the user's message is a greeting (e.g. "hi", "hello", "hey", "good morning", "good evening"):
- Politely introduce yourself.
- Mention the company name and contact information.
- Do NOT use the chunks.
- Do NOT say "I don’t know".
- Keep the response short and friendly.

NORMAL MODE:
- Answer using ONLY the provided chunks.
- If the answer is not explicitly stated, reply exactly: I don’t know.
- Do NOT add extra information or assumptions.

Company Information:
{CompanyTemplate()}

# Example
Chunks: ["Python released in 1991? 1991.", "Python popular? I don’t know."]
Question: When was Python released?
Assistant: 1991

Chunks:
[
{chunks_text}
]

User: "{self.query}"
Assistant:""".strip()


class UserIntentRephraseTemplate(Template):
    def __init__(self, last_query: str, query_history: List[ChatMsg]):
        self.last_query = last_query
        self.query_history = query_history

    def __str__(self) -> str:
        history_text = "\n".join(
            f"- {q.msg}" for q in self.query_history
        )

        return f"""
System:
You are an intent-understanding system.

Your task is to:
- Understand the user's true intent using the full conversation history.
- Rephrase the user's intent clearly and precisely.
- Do NOT answer the request.
- Do NOT add new information.
- Do NOT remove meaning.
- Return ONLY the rephrased intent as a single question.
- The returned sentence shouldn't mention the last history.

Conversation History:
{history_text}

Last User Query:
"{self.last_query}"

Assistant:
""".strip()
