import hashlib
from typing import Any, List

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from imgflw.usecase import Settings, WorkflowStore


class ChromaWorkflowStore(WorkflowStore):
    def __init__(self) -> Any:
        self.embeddings_cache = None
        self.workflow_collection = None
        self.embedding_function = None

    def get_workflow_collection(self) -> chromadb.Collection:
        if self.workflow_collection is None:
            self.embedding_function = OpenAIEmbeddingFunction(
                api_key=Settings.get("openai_api_key"),
                model_name="text-embedding-ada-002",
            )
            chroma_client = chromadb.PersistentClient()
            self.workflow_collection = chroma_client.get_or_create_collection(
                "workflow", embedding_function=self.embedding_function
            )
            self.embeddings_cache = chroma_client.get_or_create_collection(
                "embedding", embedding_function=self.embedding_function
            )
        return self.workflow_collection

    def get_embedding_cache(self) -> chromadb.Collection:
        if self.embeddings_cache is None:
            self.get_workflow_collection()
        return self.embeddings_cache

    def save(self, request: str, workflow: str) -> None:
        id = self.to_id(request)
        self.get_workflow_collection().upsert(
            ids=[id],
            documents=[request],
            metadatas=[{"workflow": workflow}],
        )

    def to_id(self, request: str) -> str:
        return hashlib.sha256(request.encode()).hexdigest()

    def get(self, request: str) -> str:
        id = self.to_id(request)
        result = self.get_workflow_collection().get(ids=[id])
        metadatas = result["metadatas"]
        if len(metadatas) == 0:
            return ""
        return metadatas[0]["workflow"]

    def find(self, request: str) -> List[str]:
        embeddings = self.get_embeddings(request)
        results = self.get_workflow_collection().query(query_embeddings=embeddings, n_results=4)
        documents = results["documents"]
        if len(documents) == 0:
            return []
        distances = [f"{d:.2f}" for d in results["distances"][0]]
        print(request, distances)
        return documents[0]

    def get_embeddings(self, request: str) -> chromadb.Embeddings:
        id = self.to_id(request)
        embedding_collection = self.get_embedding_cache()
        result = embedding_collection.get(ids=[id], include=["embeddings"])
        embeddings = result["embeddings"]
        if len(embeddings) == 0:
            embeddings = self.embedding_function([request])
            embedding_collection.add(ids=[id], embeddings=embeddings)

        return embeddings
