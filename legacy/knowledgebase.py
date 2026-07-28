# knowledgebase.py

import chromadb
from chromadb import PersistentClient
from embeddings import ChromaEmbeddingFunction

class KnowledgeBase:
    def __init__(self, collection_name="my_collection", persist_directory="chroma_db"):
        """
        Initializes Chroma client and a specific collection.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        embedding_function = ChromaEmbeddingFunction()
        self.collection_name = collection_name
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function
        )

    def add_document(self, doc_id: str, text: str, meta: dict):
        """
        Add a single document to Chroma. If doc_id already exists, remove it first.
        """
        # Check if the document exists before deleting
        existing_docs = self.collection.get(ids=[doc_id])
        
        if existing_docs["ids"]:
            self.collection.delete(ids=[doc_id])  # Delete existing entry if found
        
        # Add new document

        self.collection.add(
            documents=[text],
            metadatas=[meta],
            ids=[doc_id]
        )

    def query(self, query_text: str, n_results: int = 3):
        """
        Query the knowledge base using vector similarity search.
        Returns up to n_results documents.
        """
        # results = self.collection.query(
        #     query_texts=[query_text],
        #     n_results=n_results
        # )
        # return results
        embedding_function = ChromaEmbeddingFunction()
        query_embedding = embedding_function.embed_query(query_text)

        results = self.collection.query(
            query_embeddings=[query_embedding],  # Use embeddings instead of text
            n_results=n_results
        )

        # Check if results are empty and return a meaningful response
        if not results["documents"] or not results["documents"][0]:
            return "No relevant information found in the knowledge base."

        return results
    # def clear_collection(self):
    #     """
    #     Clears all documents from the current Chroma collection.
    #     """
    #     self.client.delete_collection(self.collection_name)  # Delete the entire collection
    #     self.collection = self.client.get_or_create_collection(  # Recreate the collection
    #         name=self.collection_name,
    #         embedding_function=ChromaEmbeddingFunction()
    #     )
    #     print("All documents have been deleted and collection reset.")



    def persist(self):
        """
        Persist Chroma data to disk (in the 'chroma_db' folder).
        """
        pass
       
