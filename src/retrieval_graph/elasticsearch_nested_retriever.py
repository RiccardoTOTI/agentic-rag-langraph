from typing import List, Union, Dict, Any
from elasticsearch import Elasticsearch
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.embeddings.base import Embeddings
import asyncio
from pydantic import PrivateAttr



class ElasticsearchNestedRetriever:
    def __init__(
        self,
        es: Elasticsearch,
        index_names: Union[str, List[str]],
        embedding_model: Embeddings,
        vector_field: str = "chunks.embedding",
        text_field: str = "chunks.text",
        nested_path: str = "chunks",
        k: int = 4
    ):
        self.es = es
        self.index_names = [index_names] if isinstance(index_names, str) else index_names
        self.embedding_model = embedding_model
        self.vector_field = vector_field
        self.text_field = text_field
        self.nested_path = nested_path
        self.k = k

    def _query_vector(self, query: str) -> List[float]:
        return self.embedding_model.embed_query(query)

    def invoke(self, query: str) -> List[Document]:
        vector = self._query_vector(query)

        query_body = {
            "size": 10,
            "query": {
                "nested": {
                    "path": self.nested_path,
                    "inner_hits": {
                        "size": self.k,
                        "_source": True
                    },
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": f"cosineSimilarity(params.query_vector, '{self.vector_field}') + 1.0",
                                "params": {"query_vector": vector}
                            }
                        }
                    }
                }
            }
        }

        response = self.es.search(index=self.index_names, body=query_body)

        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            index_name = hit["_index"]
            inner_hits = hit["inner_hits"][self.nested_path]["hits"]["hits"]
            for inner in inner_hits:
                chunk = inner["_source"]
                metadata = {
                    **chunk.get("metadata", {}),
                    "filename": source.get("filename"),
                    "path": source.get("path"),
                    "checksum": source.get("checksum"),
                    "tikaLanguage": source.get("tikaLanguage"),
                    "indexName": index_name
                }
                results.append(Document(page_content=chunk[self.text_field.split('.')[-1]], metadata=metadata))

        return results

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        return ElasticsearchNestedLangchainRetriever(self)


class ElasticsearchNestedLangchainRetriever(BaseRetriever):
    _retriever: ElasticsearchNestedRetriever = PrivateAttr()

    def __init__(self, retriever: ElasticsearchNestedRetriever):
        super().__init__()  # Pydantic BaseModel init
        object.__setattr__(self, "_retriever", retriever)  # workaround compatibile con Pydantic v1

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._retriever.invoke(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._retriever.invoke, query)


