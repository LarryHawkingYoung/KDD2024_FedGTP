import torch, copy
import torch.nn as nn
from model.AGCRNCell import AGCRNCell
import lib.utils as utils

class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.args = args

        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)
        if self.args.active_mode == "adptpolu":
            self.poly_coefficients = nn.Parameter(torch.randn(1, args.act_k+1), requires_grad=True)
        else: self.poly_coefficients = None

        self.encoder = AVWDCRNN(self.args, self.num_nodes, args.input_dim, args.rnn_units,
                                args.embed_dim, args.num_layers)

        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source):
        #source: B, T_1, N, D
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings, self.poly_coefficients)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        # output = checkpoint(self.end_conv, (output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)                          #B, T, N, C

        return output

    def comm_socket(self, msg, device=None):
        self.args.socket.send(msg)
        if device:
            return self.args.socket.recv().to(device)
        else:
            return self.args.socket.recv()
    

    def fedavg(self):
        if self.args.active_mode == "adptpolu":
            mean_p = self.comm_socket(self.poly_coefficients.data, self.args.device) / self.args.num_clients
            self.poly_coefficients = nn.Parameter(mean_p, requires_grad=True).to(mean_p.device)

        model_dict = self.comm_socket(self.end_conv.state_dict())
        self.end_conv.load_state_dict(model_dict)
        self.encoder.fedavg()


class AVWDCRNN(nn.Module):
    def __init__(self, args, node_num, dim_in, dim_out, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.args = args
        self.dcrnn_cells.append(AGCRNCell(self.args, node_num, dim_in, dim_out, embed_dim))

        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.args, node_num, dim_out, dim_out, embed_dim))

    def forward(self, x, init_state, node_embeddings, poly_coefficients):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim, (x.shape, self.node_num)
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, poly_coefficients)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states
    
    def fedavg(self):
        for model in self.dcrnn_cells: model.fedavg()