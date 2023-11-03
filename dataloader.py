import os
import sys
import numpy as np
from kg_handler import KGHandler
from data_splitter import Splitter
from interaction_file_handler import InteractionsHandler


class DataLoader:
    def __init__(self, params):
        self.params = params
        self.set_file_paths()
        
        # interactions
        self.interaction_data = InteractionsHandler(self.interaction_file)
        
        # interactions train test split
        
        # KG
        self.n_attr = 0 # taken from SW. dunno what it does
        self.num_entities, self.num_relations, self.num_triples = self.data_stats()
        self.total_entities_and_attributes = self.n_entity + self.n_attr
        self.kg_data = KGHandler(self.kg_file)


    def set_file_paths(self):
        self.kg_file = os.path.join(self.params['dataset_path'], 'train2id.txt')
        self.interaction_file = os.path.join(self.params['dataset_path'], 'inter2id.txt')
        self.entity_file = os.path.join(self.params['dataset_path'], 'entity2id.txt')
        self.relation_file = os.path.join(self.params['dataset_path'], 'relation2id.txt')
    
    def data_stats(self):
        with open(self.entity_file, 'r') as f:
            n_entity = int(f.readline().strip())
        with open(self.rel_file, 'r') as f:
            n_relation = int(f.readline().strip())
        with open(self.kg_file, 'r') as f:
            n_triple = int(f.readline().strip())
        return n_entity, n_relation, n_triple
    