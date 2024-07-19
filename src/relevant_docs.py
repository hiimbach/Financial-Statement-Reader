from haystack import component


@component
class CustomDocumentStore:
    def __init__(self, documents):
        self.documents = documents
        self.doc_store_len = len(documents)

    def run(self, founded_doc) -> str:
        # Because this uses the output of the retriever, first unpack it
        info = founded_doc['retriever']['documents'][0]
        idx = info.meta['index']

        # Get surround docs
        if idx == 0:
            return f"{self.documents[0]}\n{self.documents[1]}"
        elif idx == self.doc_store_len-1:
            return f"{self.documents[self.doc_store_len-2]}\n{self.documents[self.doc_store_len-1]}"
        else:
            return f"{self.documents[idx-1]}\n{self.documents[idx]}\n{self.documents[idx+1]}\n"

