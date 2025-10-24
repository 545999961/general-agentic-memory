import os
import json
import numpy as np
from typing import Dict, Any, List
from FlagEmbedding import FlagAutoModel
from FlagEmbedding.abc.evaluation.utils import index as faiss_index
from FlagEmbedding.abc.evaluation.utils import search

from gam.retriever.base import AbsRetriever
from gam.schemas import InMemoryPageStore, Hit


class IndexRetriever(AbsRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.page_store = None

    def search(self, query_indices: List[List[str]], top_k: int = 10) -> List[List[Hit]]:
        hits: List[Hit] = []
        for query_indice in query_indices:
            for pid in query_indice:
                p = self.page_store.get(pid)
                if not p:
                    continue
                hits.append(Hit(
                    page_id=p.page_id,
                    snippet=p.content[:200],
                    source="page_id",
                    meta={}
                ))
        return hits

    def load(self):
        index_dir = self.config.index_dir
        try:
            self.page_store = InMemoryPageStore.load(os.path.join(index_dir, "pages"))
        except Exception as e:
            print('cannot load index, error: ', e)

    def build(self, page_store: InMemoryPageStore):
        self.page_store = page_store

        self.page_store.save(os.path.join(self.config.index_dir, "pages"))
    
    def update(self, page_store: InMemoryPageStore):
        self.build(page_store)

