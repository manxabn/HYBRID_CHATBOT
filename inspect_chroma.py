import chromadb

# Path to your ChromaDB persistent storage directory
persist_directory = "chroma_db"

# Connect to the ChromaDB database
client = chromadb.PersistentClient(path=persist_directory)

# List all collections in the database
collections = client.list_collections()
print("\nAvailable collections in ChromaDB:")
for collection_name in collections:
    print(collection_name)  # collections now returns a list of names (strings)

# Allow user to select a collection to inspect
if collections:
    collection_name = collections[0]  # Select the first collection for demonstration
    collection = client.get_collection(collection_name)

    # Retrieve all stored documents and metadata
    results = collection.get(include=["embeddings", "documents", "metadatas"])
    print("\nChromaDB Collection Contents with Embeddings:")

    for i, doc in enumerate(results["documents"]):
        print(f"\nDocument {i + 1}:")
        print(f"ID: {results['ids'][i]}")
        print(f"Text: {doc}")
        print(f"Metadata: {results['metadatas'][i]}")
        print(f"Embedding: {results['embeddings'][i]}")  # Check if embeddings are stored

else:
    print("No collections found in ChromaDB.") 
