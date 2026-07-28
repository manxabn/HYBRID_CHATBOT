# main.py

import os
from create_and_populate_db import create_and_populate_db
from db_ingestion import DBIngestion
from agent import create_llm_agent

def main():
    # (1) Create and populate the SQLite database (only needed once).
    #     If you've already populated the DB, you can comment this out.
    #create_and_populate_db()

    # (2) Ingest data from DB into Chroma (incremental).
    ingestor = DBIngestion(db_path="knowledge_base.db")
    ingestor.run_ingestion()

    # (3) Create the LLM agent using the knowledge base in memory
    kb = ingestor.kb  # Our KnowledgeBase object
    agent = create_llm_agent(kb)

    print("=== BRAC University Chatbot ===")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        response = agent.invoke({"input": user_input})
        print(f"Assistant: {response['output']}\n")

if __name__ == "__main__":
    main()
# from knowledgebase import KnowledgeBase

# def clear_chroma_data():
#     kb = KnowledgeBase()
#     kb.clear_collection()
#     print("ChromaDB collection has been cleared successfully.")

# if __name__ == "__main__":
#     clear_chroma_data()
