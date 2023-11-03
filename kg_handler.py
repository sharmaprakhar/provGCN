import os
import sys
import numpy as np
from collections import defaultdict


class KGHandler:
    def __init__(self, kg_file) -> None:
        self.kg_file = kg_file
        self.kg_stats()

    def load_kg_data(self):
        kg_triple_mat = np.loadtxt(self.kg_file, dtype=np.int64, skiprows=1)
        kg_triples = np.unique(kg_triple_mat, axis=0)
        self.relations_dict = defaultdict(list)
        for triple in kg_triples:
            head, tail, r = triple
            self.relations_dict[r].append((head, tail)) # is tuple, maybe list?