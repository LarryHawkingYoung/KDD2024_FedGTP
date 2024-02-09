import torch, copy
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import itertools
import lib.utils as utils

class AVWGCN(nn.Module):
    def __init__(self, args, dim_in, dim_out, embed_dim):
        super(AVWGCN, self).__init__()
        self.args = args
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings, poly_coefficients):
        # node_embeddings: n d
        # x: b n c
        assert self.args.active_mode in ['sprtrelu', 'adptpolu']

        transformed_E, EH = self.gen_upload(x, node_embeddings)

        if self.args.active_mode == "sprtrelu":
            sum_EH = self.comm_socket(EH, self.args.device)
            return self.recv_fwd(node_embeddings, x, transformed_E, sum_EH)

        elif self.args.active_mode == "adptpolu":
            sum_EH = [self.comm_socket(eh, self.args.device) for eh in EH]
            return self.recv_fwd(node_embeddings, x, transformed_E, sum_EH, poly_coefficients)

    def gen_upload(self, x, node_embeddings):
        E = node_embeddings # n d
        H = x # b n c
        if self.args.active_mode == "sprtrelu":
            transformed_E = torch.relu(E)
            EH = torch.einsum("dn,bnc->bdc", transformed_E.transpose(0,1), H)
        elif self.args.active_mode == "adptpolu":
            transformed_E = [self.transform(k, E) for k in range(self.args.act_k+1)]
            EH = [torch.einsum("dn,bnc->bdc", e.transpose(0,1), H) for e in transformed_E]
        return transformed_E, EH

    def recv_fwd(self, E, H, transformed_E, sum_EH, P=None):
        if self.args.active_mode == "sprtrelu":
            Z = H + torch.einsum("nd,bdc->bnc", transformed_E, sum_EH)
        elif self.args.active_mode == "adptpolu":
            Z = torch.stack([torch.einsum("nd,bdc->bnc", transformed_E[i], sum_EH[i]) for i in range(self.args.act_k+1)])
            Z = torch.einsum('ak,kbnc->abnc', P, Z)[0]
            Z = H + Z

        weights = torch.einsum('nd,dio->nio', E, self.weights_pool)  #N, dim_in, dim_out
        bias = torch.matmul(E, self.bias_pool)                       #N, dim_out
        x_gconv = torch.einsum('bni,nio->bno', Z, weights) + bias     #b, N, dim_out
        return x_gconv
    
    def comm_socket(self, msg, device=None):
        self.args.socket.send(msg)
        if device:
            return self.args.socket.recv().to(device)
        else:
            return self.args.socket.recv()

    def cartesian_prod(self, A, B):
        transformed = torch.stack(list(map(torch.cartesian_prod, A, B)))
        transformed = transformed[...,0] * transformed[...,1]
        return transformed

    def transform(self, k, E):
        ori_k = k
        transformed = torch.ones(E.shape[0], 1).to(E.device)
        cur_pow = self.cartesian_prod(transformed, E)
        while k > 0:
            if k % 2 == 1:
                transformed = self.cartesian_prod(transformed, cur_pow)
            cur_pow = self.cartesian_prod(cur_pow, cur_pow)
            k //= 2
        assert transformed.shape[0] == E.shape[0], (transformed.shape[0], E.shape[0])
        assert transformed.shape[1] == E.shape[1]**ori_k, (transformed.shape[1], E.shape[1], ori_k)
        return transformed

    def fedavg(self):
        mean_w = self.comm_socket(self.weights_pool.data, self.args.device) / self.args.num_clients
        mean_b = self.comm_socket(self.bias_pool.data, self.args.device) / self.args.num_clients

        self.weights_pool = nn.Parameter(mean_w, requires_grad=True).to(mean_w.device)
        self.bias_pool = nn.Parameter(mean_b, requires_grad=True).to(mean_b.device)

