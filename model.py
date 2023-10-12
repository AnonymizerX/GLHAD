
import torch

import config as c
import utils.utils as u
import torch.nn.functional as F

from torch import Tensor, nn
from torch_sparse import SparseTensor, matmul
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing, HeteroConv



class LocalHomo(MessagePassing):
    def __init__(self, in_channels, hidden_channels):
        super().__init__(aggr="mean")
        # self.lin=nn.Linear(in_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(2*in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

        self.pre_edge_type = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp[0].weight.data)
        nn.init.xavier_uniform_(self.mlp[2].weight.data)

    def forward(self, x, edge_index):

        pre_local_homo = self.propagate(edge_index=edge_index, x=x)

        return pre_local_homo, self.pre_edge_type

    def message(self, x_i, x_j):
        self.pre_edge_type = self.mlp(torch.cat([x_i, x_j], dim=-1))
        return self.pre_edge_type


class GLHAD_Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr, parameter_matrix, filter_type=0):
        super(GLHAD_Conv, self).__init__(aggr=aggr)
        self.parameter_matrix = parameter_matrix
        self.filter_type = filter_type

        if self.parameter_matrix:
            if self.filter_type == 'dis':
                self.low_W = Parameter(
                    torch.zeros([in_channels, out_channels]))
                self.high_W = Parameter(
                    torch.zeros([in_channels, out_channels]))
            elif self.filter_type == 'mix':
                self.W = Parameter(torch.zeros([in_channels, out_channels]))
            self.reset_parameters()

    def reset_parameters(self):
        if self.filter_type == 'dis':
            nn.init.xavier_uniform_(self.low_W.data)
            nn.init.xavier_uniform_(self.high_W.data)
        elif self.filter_type == 'mix':
            nn.init.xavier_uniform_(self.W.data)

    def forward(self, x, filter_l, filter_h, local_homo):
        if self.filter_type == 'dis':
            x_L = self.propagate(
                edge_index=filter_l, x=x)
            x_H = self.propagate(
                edge_index=filter_h, x=x)

            if self.parameter_matrix:
                x_L = torch.mm(x_L, self.low_W)
                x_H = torch.mm(x_H, self.high_W)

            x_L = F.normalize(x_L, p=2, dim=-1)
            x_H = F.normalize(x_H, p=2, dim=-1)

            x_L = F.relu(x_L)
            x_H = F.relu(x_H)
            x_temp = local_homo*x_L + (1-local_homo)*x_H

        elif self.filter_type == 'mix':
            homo_filter = local_homo*filter_l+(1-local_homo)*filter_h
            x_temp = self.propagate(
                edge_index=homo_filter, x=x)

            if self.parameter_matrix:
                x_temp = torch.mm(x_temp, self.W)

            x_temp = F.normalize(x_temp, p=2, dim=-1)
            x_temp = F.relu(x_temp)

        return x_temp

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)


class GLHAD(nn.Module):
    """local homo graph neural network"""

    def __init__(
        self,
        in_channels,
        out_channels,
        config,
        num_nodes,
        ds='cora',
        edge_index=None,
        device=None
    ):
        super(GLHAD, self).__init__()

        hidden_channels = config['hidden_channels']
        self.ds = ds
        self.K = config['K']
        self.beta = config['beta']
        self.dp = config['dropout']
        self.config = config
        self.homo = config['homo']
        self.filter_name = config['filter_name']

        if self.homo:
            self.filter_l, self.filter_h = u.cal_low_high_filter(
                dev=device, N=num_nodes, beta=config['beta'], dataset=self.ds, edge_name='Homo', edge_index=edge_index, filter_name=self.filter_name, add_self_loop=config['self_loop'])

        else:
            self.filter_l = {}
            self.filter_h = {}
            for curr_edge_name, curr_edge_index in edge_index.items():
                self.filter_l[curr_edge_name], self.filter_h[curr_edge_name] = u.cal_low_high_filter(
                    dev=device, N=num_nodes, beta=config['beta'], dataset=self.ds, edge_name=curr_edge_name, edge_index=curr_edge_index, filter_name=self.filter_name, add_self_loop=config['self_loop'])

        self.local_homo = LocalHomo(
            in_channels=in_channels, hidden_channels=hidden_channels)

        self.convs = nn.Sequential(GLHAD_Conv(
            in_channels=in_channels, out_channels=hidden_channels, aggr='add', parameter_matrix=config['parameter_matrix'], filter_type=config['filter_type']))
        for _ in range(1, self.K):
            self.convs.append(GLHAD_Conv(
                in_channels=hidden_channels, out_channels=hidden_channels, aggr='add', parameter_matrix=config['parameter_matrix'], filter_type=config['filter_type']))

        if config['A_embed']:
            self.A_mlp = nn.Sequential(
                nn.Linear(num_nodes, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
            )
        else:
            self.A_mlp = None

        self.multi_layer_concate = config['multi_layer_concate']

        if config['out_mlp']:
            if config["parameter_matrix"]:
                self.out_linear = nn.Sequential(
                    nn.Linear(hidden_channels if not self.multi_layer_concate else hidden_channels *
                              self.K+in_channels, 2*hidden_channels),
                    nn.BatchNorm1d(2*hidden_channels),
                    nn.ReLU(),
                    nn.Linear(2*hidden_channels, out_channels)
                )
            else:
                self.out_linear = nn.Sequential(
                    nn.Linear(in_channels if not self.multi_layer_concate else in_channels *
                              self.K+in_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, out_channels)
                )
        else:
            self.out_linear = nn.Linear(
                hidden_channels if not self.multi_layer_concate else hidden_channels *
                self.K+in_channels, out_channels)

    def forward(self, x, edge_index, local_homo=None):
        N, _ = x.shape
        dev = x.device

        if self.multi_layer_concate:
            out = x

        if self.homo:
            x_temp = x.clone().detach()

            if local_homo is None and self.local_homo != None:

                local_homo, pre_edge_type = self.local_homo(x, edge_index)
            else:
                pre_edge_type = None

            for layer_index, conv in enumerate(self.convs):
                x_temp = conv(x_temp, self.filter_l, self.filter_h, local_homo)

                if self.multi_layer_concate:
                    out = torch.cat((out, x_temp), dim=1)
                else:
                    # only last layer
                    if layer_index == len(self.convs)-1:
                        out = x_temp

        else:
            local_homo = {}
            pre_edge_type = {}
            for curr_edge_name, curr_edge_index in edge_index.items():
                local_homo[curr_edge_name], pre_edge_type[curr_edge_name] = self.local_homo(
                    x, curr_edge_index)

            x_temp_dict = {}
            for key in edge_index.keys():
                x_temp_dict[key] = x.clone().detach()

            for layer_index, conv in enumerate(self.convs):
                for curr_edge_name, curr_edge_index in edge_index.items():
                    x_temp_dict[curr_edge_name] = conv(
                        x_temp_dict[curr_edge_name], self.filter_l[curr_edge_name],  self.filter_h[curr_edge_name], local_homo[curr_edge_name])

                if self.multi_layer_concate:
                    x_temp = torch.stack([x_temp_dict[key]
                                         for key in edge_index.keys()]).sum(0)
                    # x_temp = torch.stack([x_temp_dict[key]
                    #                      for key in edge_index.keys()]).mean(0)
                    out = torch.cat((out, x_temp), dim=1)

                else:
                    # only last layer result
                    if layer_index == len(self.convs)-1:
                        x_all = []
                        for key in edge_index.keys():
                            x_all.append(x_temp_dict[key])
                        x_temp = torch.stack(x_all).sum(0)
                        out = x_temp

        # embedding A and concat representations
        if self.A_mlp is not None:
            A = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                value=torch.ones([edge_index.shape[1]]).to(dev),
                sparse_sizes=[N, N]).to_torch_sparse_coo_tensor()
            A = self.A_mlp(A)
            out = torch.cat([out, A], dim=-1)
        else:
            out = out

        # norm (K+1, N, hdim)
        if self.config['out_norm']:
            out = F.normalize(out, p=2, dim=-1)
        out = F.dropout(out, self.dp, self.training)

        # prediction
        out = self.out_linear(out)
        return out, local_homo, pre_edge_type
