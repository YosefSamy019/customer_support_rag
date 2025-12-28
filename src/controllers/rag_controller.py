from typing import List

from src.models.models import Chunk, ChatMsg, ChatSender
from src.models.project_model import Project
from src.templates.template_interface import QueryChunksAnsweringTemplate, UserIntentRephraseTemplate


class RAGController:
    @staticmethod
    def storage_add_pdf(project: Project, pdf_path: str) -> None:
        text: str = project.ingestion_controller.ingest_from_pdf(pdf_path)
        chunks: List[Chunk] = project.chunks_controller.fixed_window_slicer(text)
        chunks: List[Chunk] = project.vector_db.filter_new_chunks(chunks)
        chunks: List[Chunk] = project.sentence_llm.call_chunks(chunks, batch_size=25)
        project.vector_db.store_chunks(chunks)

    @staticmethod
    def storage_add_json(project: Project, json_path: str) -> None:
        texts: List[str] = project.ingestion_controller.ingest_from_json(json_path)
        chunks: List[Chunk] = project.chunks_controller.no_slice(texts)
        chunks: List[Chunk] = project.vector_db.filter_new_chunks(chunks)
        chunks: List[Chunk] = project.sentence_llm.call_chunks(chunks, batch_size=25)
        project.vector_db.store_chunks(chunks)

    @staticmethod
    def search_with_query(project: Project, query: str) -> dict:
        user_query = str(query)

        project.chat_history.append(ChatMsg(msg=user_query, metadata=None, sender=ChatSender.user))

        query_history: List[ChatMsg] = list(filter(lambda x: x.sender == ChatSender.user, project.chat_history))

        # Intent Rephrase
        if len(query_history) > 1:
            intent_rephrase_template = UserIntentRephraseTemplate(
                last_query=user_query,
                query_history=query_history
            )

            rephrased_user_query = project.intent_rephraser_llm.call(intent_rephrase_template)
        else:
            rephrased_user_query = user_query

        user_query_embedding = project.sentence_llm.call([rephrased_user_query])[0]

        candidate_chunks: List[Chunk] = project.vector_db.search_embedding(user_query_embedding)

        candidate_chunks_json: List[dict] = list(map(lambda c: c.to_dict(), candidate_chunks))

        llm_input_prompt: str = QueryChunksAnsweringTemplate(query, candidate_chunks).as_str()

        llm_response: str = project.QA_llm.call(llm_input_prompt)

        return_dict = {
            "original_query": query,
            "rephrased_user_query": rephrased_user_query,
            "project_name": project.project_name,
            "llm_input_prompt": llm_input_prompt,
            "response": llm_response,
            "candidate_chunks": candidate_chunks_json,
        }

        project.chat_history.append(ChatMsg(msg=llm_response, metadata=return_dict, sender=ChatSender.assistant))

        return return_dict
