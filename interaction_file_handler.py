import os
import sys
import numpy as np
from collections import defaultdict


class InteractionsHandler:
    def __init__(self, interaction_file) -> None:
        self.interaction_file = interaction_file
        self.interaction_matrix, self.interaction_dict, self.num_total_interactions = self.parse_interaction_files()
        self.entites_list = list(self.interaction_dict.keys())
        self.total_entities = len(self.interaction_dict)

    # load_ratings method in SW
    # an interaction is a pair (start_node, end_node)
    def parse_interaction_files(self):
        interaction_matrix = set()
        interaction_dict = defaultdict(set)

        with open(self.interaction_file, 'r') as fr:
            line = fr.readline()
            while line:
                line = line.strip.split(' ')
                line = list(map(int, line))
                start_node, end_nodes = line[0], line[1:]
                for node in end_nodes:
                    interaction_dict[start_node].add(node)
                    interaction_matrix.add((start_node, node))
        # convert set of tuples to np array
        interaction_matrix = np.array([[item for item in pair] for pair in interaction_matrix])
        num_total_interactions = interaction_matrix.shape[0]

        return interaction_matrix, interaction_dict, num_total_interactions