import numpy as np
import torch
import scipy.sparse as sp
from dataloader import DataLoader
from collections import defaultdict

class GCNData(DataLoader):
    def __init__(self, params):
        super().__init__(params)
        self.adj_norm_scheme = params['adj_norm_scheme']
        self.make_adj_matrices_list()
        self.normalize_adj_matrices_list()
        # no reorder for sparse tensors required


    def create_KG_dict(self):
        self.master_h_to_tr_mapping = defaultdict(list)
        for normalized_adj_matrix in self.norm_adj_matrices:
            rows = normalized_adj_matrix.row
            cols = normalized_adj_matrix.col
            for i in range(len(rows)):
                head = rows[i]
                tail = cols[i]
                relation = self.adj_relations_list[i] # defined for each KG triple
                self.master_h_to_tr_mapping[head].append((tail, relation))

        self.heads_list = list(self.master_h_to_tr_mapping.keys())
        self.total_heads = len(self.heads_list)


    def normalized_adj_matrices(self):
        '''
        This is the KGAT implementation based on symmetric and random walk
        laplacian matrices. They also call them bi normalized or si-norm adj matrices
        '''
        def symmetric_norm(adj_mat):
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt) # GCN constant
            return norm.tocoo()
        
        def rand_walk_norm(adj_mat):
            # maybe a diff thing than KGAT?
            rowsum = np.array(adj_mat.sum(axis=1))
            # It is reasonable for np.power(rowsum, -1).flatten() to trigger divide by zero encountered warning
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            return norm_adj.tocoo()

        if self.adj_norm_scheme=='random_walk':
            self.norm_adj_matrices = [rand_walk_norm(adj) for adj in self.all_adj_matrices]
        elif self.adj_norm_scheme=='symmetric':
            self.norm_adj_matrices = [symmetric_norm(adj) for adj in self.all_adj_matrices]



    def make_sparse_adj_list(self, mat, row_pre=0, col_pre=0):
        '''
        create sparse adjacency lists 
        # n_all = self.n_entity_attr - from SW. why declare n_attr??
        '''
        sparse_mat_shape = self.total_entities_and_attributes
        # to-node interaction: A: A->B
        a_rows = mat[:, 0] + row_pre
        a_cols = mat[:, 1] + col_pre
        # must use float 1. (int 1 is not allowed)
        a_vals = [1.] * len(a_rows)

        # from-node interaction: A: B->A
        b_rows = a_cols
        b_cols = a_rows
        b_vals = [1.] * len(b_rows)

        a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(sparse_mat_shape, sparse_mat_shape))
        b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(sparse_mat_shape, sparse_mat_shape))

        return a_adj, b_adj

    def get_relational_adj_list(self, add_inv=0):
        '''
        create adjacency lists for interaction data and knowledge graph
        The reverse direction adj matrices are disabled right now, turn on by 
        uncommenting the inv code

        # n_all = self.n_entity_attr - from SW. why declare n_attr??
        '''
        
        self.adj_mat_list = []
        self.adj_relations_list = []
        
        #handle interactions
        interaction_adj, inv_interaction_adj = self.make_sparse_adj_list(self.interaction_data.interaction_matrix)
        self.adj_mat_list.append(interaction_adj)
        self.adj_relations_list.append(0) # because direct interactions are 0
        if add_inv:
            self.adj_mat_list.append(inv_interaction_adj)
            self.adj_relations_list.append(1)
        #handle KG
        for r,mat in self.kg_data.relations_dict.items():
            r_adj, inv_r_adj = self.make_sparse_adj_list(mat)
            self.adj_mat_list.append(r_adj)
            self.adj_relations_list.append(r+1)
            if add_inv:
                self.adj_mat_list.append(inv_r_adj)
                self.adj_relations_list.append(r+1 + self.num_relations+1)