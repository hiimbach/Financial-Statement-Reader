import os
from typing import List

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

from src.relevant_docs import CustomDocumentStore


os.environ["GOOGLE_API_KEY"] = "AIzaSyAAAR-c4ahbKgIZmRv-6zBUZWWAyJrEHqI"
TEMPLATE = """
"""


class RAGPipeline:
    def __init__(self, documents: List[str], prompt_template: str):
        # Create document store and embedder
        self.document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        formatted_docs = [Document(content=doc, meta={"index": i}) for i, doc in enumerate(documents)]

        self.document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        self.document_embedder.warm_up()
        documents_with_embeddings = self.document_embedder.run(formatted_docs)["documents"]
        self.document_store.write_documents(documents_with_embeddings)

        # Pipeline components
        self.text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        self.custom_document_store = CustomDocumentStore(documents)
        self.prompt_builder = PromptBuilder(template=prompt_template)
        self.generator = GoogleAIGeminiGenerator(model='gemini-pro')

        # Create pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("text_embedder", self.text_embedder)
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("custom_document_store", self.custom_document_store)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.generator)

        # Connect components together
        self.pipeline.connect("text_embedder", "retriever.query_embedding")
        self.pipeline.connect("retriever", "custom_document_store")
        self.pipeline.connect("custom_document_store", "prompt_builder.query")
        self.pipeline.connect("prompt_builder", "llm")

    def run(self, query: str) -> str:
        return self.pipeline.run({
            "text_embedder": {
                "text": query
            }
        })['llm']['replies'][0]


class InfoExtractor:
    def __init__(self, template: str):
        self.pipeline = Pipeline()
        self.prompt_builder = PromptBuilder(template=template)
        self.generator = GoogleAIGeminiGenerator(model='gemini-pro')
        
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.generator)
        self.pipeline.connect("prompt_builder", "llm")

    def run(self, query: str) -> str:
        return self.pipeline.run({
            "prompt_buider": {
                "query": query
            }
        })['llm']['replies'][0]
