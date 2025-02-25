from typing import List

from haystack import Document
from haystack import component


@component
class CustomDocumentStore:
    def __init__(self, documents, reference_docs):
        self.documents = documents
        self.ref_docs = reference_docs
        self.doc_store_len = len(documents)

    @component.output_types(full_context=str)
    def run(self, founded_doc: List[Document]) -> str:
        # Because this uses the output of the retriever, first unpack it

        full_context = ''
        idx_list = []
        for i, doc in enumerate(founded_doc):
            idx = founded_doc[i].meta['index']
            print(f"Index: {idx}, Score: {founded_doc[i].score}")
            # Get surround docs
            idx_list.append(idx)
            idx_list.append(idx+1)

        idx_list = sorted(list(set([i for i in idx_list if i < self.doc_store_len])))
        print("Indexes used: ", idx_list)

        for idx in idx_list:
            if idx == self.doc_store_len-1:
                full_context += f"{self.ref_docs[self.doc_store_len-2]}\n{self.ref_docs[self.doc_store_len-1]}"
            else:
                full_context += f"{self.ref_docs[idx]}\n{self.ref_docs[idx+1]}\n"

        return {"full_context": full_context}