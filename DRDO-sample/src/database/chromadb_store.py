"""
Vector Database using ChromaDB for storing embeddings.
"""

import chromadb
from chromadb.config import Settings

class ChromaDBStore:
    def __init__(self, collection_name='anomaly_embeddings'):
        self.client = chromadb.Client(Settings())
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_embedding(self, embedding, metadata=None):
        """
        Add an embedding to the database.
        """
        id = str(len(self.collection.get()['ids']) + 1)
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata] if metadata else None,
            ids=[id]
        )

    def query_similar(self, embedding, n_results=5):
        """
        Query similar embeddings.
        """
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return results