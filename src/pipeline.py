import os
from typing import List

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack.components.embedders import AzureOpenAITextEmbedder, AzureOpenAIDocumentEmbedder
from haystack.components.generators import AzureOpenAIGenerator

from src.relevant_docs import CustomDocumentStore


class DefineKey:
    def __init__(self):
        self.key_list = ["AIzaSyAAAR-c4ahbKgIZmRv-6zBUZWWAyJrEHqI",
                         "AIzaSyA6VHGEzZIkfAvO8fPZJjeY5eo9WFwWpWQ"]
        self.idx = 0
        os.environ["GOOGLE_API_KEY"] = self.key_list[self.idx]
        os.environ["AZURE_OPENAI_API_KEY"] = "add yours"
    def change(self):
        if self.idx == 0:
            self.idx = 1
        else:
            self.idx = 0
        os.environ["GOOGLE_API_KEY"] = self.key_list[self.idx]


class RAGPipeline:
    def __init__(self,
                 documents: List[str],
                 reference_docs: List[str],
                 prompt_template: str):
        # Create document store and embedder
        self.document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        formatted_docs = [Document(content=doc, meta={"index": i}) for i, doc in enumerate(documents)]

        # model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_name = 'BAAI/bge-m3'
        self.document_embedder = SentenceTransformersDocumentEmbedder(model=model_name)
        self.document_embedder.warm_up()
        documents_with_embeddings = self.document_embedder.run(formatted_docs)["documents"]
        self.document_store.write_documents(documents_with_embeddings)

        # Pipeline components
        self.text_embedder = SentenceTransformersTextEmbedder(model=model_name)
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store, top_k=2)
        self.custom_document_store = CustomDocumentStore(documents, reference_docs)
        self.prompt_builder = PromptBuilder(template=prompt_template)   # template should contain {query} and {context}
        self.generator = AzureOpenAIGenerator(azure_endpoint="to be added", azure_deployment="to be added")
        

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
        self.pipeline.connect("custom_document_store", "prompt_builder.context")
        self.pipeline.connect("prompt_builder", "llm")

    def run(self, query: str) -> str:
        return self.pipeline.run({
            "text_embedder": {
                "text": query
            },
            "prompt_builder": {
                "query": query
            }
        })['llm']['replies'][0]


class LLMPipeline:
    def __init__(self,
                 template: str,
                 azure_openai_key=None):
        self.pipeline = Pipeline()
        self.prompt_builder = PromptBuilder(template=template)
        if azure_openai_key:
            # HOANG OI MODIFY THIS PART NHE
            self.generator = AzureOpenAIGenerator(azure_endpoint="to be added", azure_deployment="to be added")
        
        else:
            self.generator = GoogleAIGeminiGenerator(model='gemini-pro')

        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.generator)
        self.pipeline.connect("prompt_builder", "llm")

    def run(self, query: str) -> str:
        try:
            return self.pipeline.run({
                "prompt_builder": {
                    "query": query
                }
            })['llm']['replies'][0]
        except:
            return ""
