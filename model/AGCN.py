import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

class AVWGCN(nn.Module):
    def __init__(self, args, dim_in, dim_out, embed_dim):
        super(AVWGCN, self).__init__()
        self.args = args
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        if self.args.exp_mode == 'SGL':
            total_nodes = self.args.num_nodes
            self.mask = torch.zeros(total_nodes, total_nodes).to(self.args.device)
            partitions = self.args.nodes_per
            for part in partitions:
                for i in range(len(part)):
                    for j in range(i, len(part)):
                        self.mask[part[i]][part[j]] = 1.0
                        self.mask[part[j]][part[i]] = 1.0


    def forward(self, x, node_embeddings, poly_coefficients):
        assert poly_coefficients == None
        assert self.args.active_mode in ['softmax', 'sprtrelu']
        
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]

        if self.args.active_mode == 'softmax':
            supports = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))
            supports = F.softmax(F.relu(supports), dim=1)
        elif self.args.active_mode == 'sprtrelu':
            # supports = torch.mm(F.relu(node_embeddings), F.relu(node_embeddings.transpose(0, 1)))
            supports = torch.mm(F.softmax(F.relu(node_embeddings), dim=1), F.softmax(F.relu(node_embeddings.transpose(0, 1)), dim=1))

        support_set = [torch.eye(node_num).to(supports.device), supports]
        supports = sum(support_set) #N, N

        if self.args.exp_mode == 'SGL':
            supports = supports * self.mask
            # self.args.logger.info("apply mask!!!!!!")

        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  #N, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("nm,bmc->bnc", supports, x)      #B, N, dim_in
        x_gconv = torch.einsum('bni,nio->bno', x_g, weights) + bias     #B, N, dim_out
        return x_gconv