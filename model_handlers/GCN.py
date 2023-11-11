import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import *


class ShadeWatcherGNN(nn.Module):
    def __init__(self, 
                 layer_dimensions, 
                 num_entities, 
                 num_relations, 
                 entity_dim, 
                 relation_dim,
                 reg_lambda,
                 device,
                 A_in) -> None:
        
        super(ShadeWatcherGNN).__init__()
        num_layers = len(layer_dimensions)
        self.reg_lambda = reg_lambda
        self.mean_norm = lambda x: torch.mean(x.norm(dim=1, p=2))
        
        # init Embedding layers
        self.entity_embeddings = nn.Embedding(num_entities, entity_dim)
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)
        self.transformation_M = nn.Parameter(torch.Tensor(num_relations, entity_dim, relation_dim))
        
        # init TransR params
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.linear_transform.weight)
    
    def transR_fn(self, e_h, e_r, e_t):
        # plausibility score/energy score
        v = e_h + e_r - e_t
        v.norm(dim=1, p=2) # (batch_size, 1)
        return torch.pow(v, 2)

    def calc_kg_loss(self, h, r, t, t_prime):
        # all args are arrays of indices
        r_emb = self.relation_embeddings(r) # e_r
        h_emb = self.entity_embeddings(h) # e_h
        t_emb = self.entity_embeddings(t) # e_t
        t_prime_emb = self.entity_embeddings(t_prime) # e_t_prime

        W_r = self.transformation_M[r] # (batch_size x entity_emb x relation_emb)

        # transform to relation emb space
        h_emb_r = torch.bmm(h_emb.unsqueeze(1), W_r).squeeze(1) # (batch_size x entity_emb)
        t_emb_r = torch.bmm(t_emb.unsqueeze(1), W_r).squeeze(1)
        t_prime_emb_r = torch.bmm(t_prime_emb.unsqueeze(1), W_r).squeeze(1)

        g_hrt = self.transR_fn(h_emb_r, r_emb, t_emb_r) # (batch_size, 1)
        g_hrt_prime = self.transR_fn(h_emb_r, r_emb, t_prime_emb_r) # (batch_size, 1)

        loss = torch.mean( (-1) * F.logsigmoid(g_hrt_prime - g_hrt) )
        
        # using embedding params for regularization
        reg = self.mean_norm(r_emb) + self.mean_norm(h_emb) + self.mean_norm(t_emb) + self.mean_norm(t_prime_emb)
        loss = loss + self.reg_lambda*reg
        return loss
    
    def calc_attn(self, h, t, r):
        r_emb = self.relation_embeddings(r) # e_r (1 x r_dim)
        h_emb = self.entity_embeddings(h) # e_h (batch x ent_emb_dim)
        t_emb = self.entity_embeddings(t) # e_t (batch x ent_emb_dim)
        W_r = self.transformation_M[r] # (1 x ent_emb_dim x r_dim)

        attn_coeff = (t_emb@W_r) * torch.tanh(h_emb@W_r + r_emb) # (batch, 1)

        return attn_coeff


    def update_adj_mat_with_attn_coeff(self, h_list, t_list, r_list, relations):
        # used to make sparse tensor
        rows, cols, values = [],[],[]
        
        for r in relations:
            target_indices = torch.where(r_list==r)
            target_h = h_list[target_indices]
            target_t = t_list[target_indices]
            attn_coeffs = self.calc_attn(target_h, target_t, r)

            rows.append(target_h)
            cols.append(target_t)
            values.append(attn_coeffs)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)
        # add the rest



    def forward(self, *input, mode):
        if mode=='transR':
            pass
        if mode=='cf':
            pass