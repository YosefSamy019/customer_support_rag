from src.controllers.rag_controller import RAGController
from src.models.project_model import Project

from src.loggings.loggings import *
import streamlit as st


# https://huggingface.co/datasets/MakTek/Customer_support_faqs_dataset/viewer?views%5B%5D=train

def main():
    print("Setting up...")

    ui()

def ui():
    st.set_page_config(page_title="Customer Support RAG", layout="centered")
    st.title("📞 Customer Support RAG")

    # ---- Sidebar (Side Drawer) ----
    st.sidebar.title("🧪 Debug Panel")

    show_raw = st.sidebar.checkbox("Show raw RAG result")

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if show_raw:
        st.sidebar.subheader("Last RAG Result")
        if st.session_state.last_result is not None:
            st.sidebar.write(st.session_state.last_result)
        else:
            st.sidebar.info("No query executed yet.")

    # ---- Project init ----
    if "project" not in st.session_state:
        st.session_state.project = Project(
            project_name='qa customer support',
            vector_db_path=r'assets/vectors',
        )

    # ---- Chat history ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- Chat input ----
    user_input = st.chat_input("Ask a question...")

    if user_input:
        # User message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = RAGController.search_with_query(
                    st.session_state.project,
                    user_input
                )

                # Store raw result for sidebar
                st.session_state.last_result = result

                response = result.get("response", "I don’t know.")
                st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

if __name__ == "__main__":
    main()
