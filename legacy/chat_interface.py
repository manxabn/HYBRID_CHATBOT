# chat_interface.py

import streamlit as st
from create_and_populate_db import create_and_populate_db
from db_ingestion import DBIngestion
from agent import create_llm_agent

def main():
    st.title("Agentic RAG Chatbot (University)")

    # Run only once or as needed
    #create_and_populate_db()

    # Ingest data
    ingestor = DBIngestion(db_path="knowledge_base.db")
    ingestor.run_ingestion()

    # Create the agent
    kb = ingestor.kb
    agent = create_llm_agent(kb)

    user_query = st.text_input("Ask your question:")
    if st.button("Send") and user_query:
        with st.spinner("Thinking..."):
            response = agent.run(user_query)
        st.write(response)

if __name__ == "__main__":
    main()
