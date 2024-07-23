from typing import List

from haystack import Document
from haystack import component


@component
class CustomDocumentStore:
    def __init__(self, documents):
        self.documents = documents
        self.doc_store_len = len(documents)

    @component.output_types(full_context=str)
    def run(self, founded_doc: List[Document]) -> str:
        # Because this uses the output of the retriever, first unpack it
        idx = founded_doc[0].meta['index']
        import ipdb; ipdb.set_trace()

        # Get surround docs
        if idx == 0:
            full_context = f"{self.documents[0]}\n{self.documents[1]}"
        elif idx == self.doc_store_len-1:
            full_context = f"{self.documents[self.doc_store_len-2]}\n{self.documents[self.doc_store_len-1]}"
        else:
            full_context = f"{self.documents[idx-1]}\n{self.documents[idx]}\n{self.documents[idx+1]}\n"

        import ipdb; ipdb.set_trace()

        return {"full_context": full_context}