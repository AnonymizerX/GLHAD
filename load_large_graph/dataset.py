import torch
import scipy
import numpy as np

from dgl.data.utils import load_graphs
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import add_remaining_self_loops 



def load_Amazon_Dataset(homo=1,self_loop=0):
    datapath = "./datasets/Amazon.mat"
    fulldata = scipy.io.loadmat(datapath)
    label = torch.from_numpy(fulldata['label'].flatten()).long()
    node_feat = torch.tensor(fulldata['features'].todense(), dtype=torch.float)

    data = HeteroData()
    data["node"].x = node_feat
    data['node'].y = label
    npn = torch.tensor(np.array(fulldata['net_upu'].nonzero()), dtype=torch.long)
    nsn = torch.tensor(np.array(fulldata['net_usu'].nonzero()), dtype=torch.long)
    nvn = torch.tensor(np.array(fulldata['net_uvu'].nonzero()), dtype=torch.long)

    if self_loop and not homo:
        npn, _ = add_remaining_self_loops(edge_index=npn)
        nsn, _ = add_remaining_self_loops(edge_index=nsn)
        nvn, _ = add_remaining_self_loops(edge_index=nvn)

    data["node", 'p', "node"].edge_index = npn
    data["node", 's', "node"].edge_index = nsn
    data["node", 'v', "node"].edge_index = nvn

    if homo:
        data = data.to_homogeneous()

    if self_loop and homo:
        data.edge_index, _ = add_remaining_self_loops(data.edge_index)

    return [data]


def load_Yelp_Dataset(homo=1,self_loop=0):

    datapath = "./datasets/YelpChi.mat"
    fulldata = scipy.io.loadmat(datapath)
    label = torch.from_numpy(fulldata['label'].flatten()).long()
    node_feat = torch.tensor(fulldata['features'].todense(), dtype=torch.float)

    data = HeteroData()
    data['node'].x = node_feat
    data['node'].y = label
    nun = torch.tensor(np.array(fulldata['net_rur'].nonzero()), dtype=torch.long)
    nsn = torch.tensor(np.array(fulldata['net_rsr'].nonzero()), dtype=torch.long)
    ntn = torch.tensor(np.array(fulldata['net_rtr'].nonzero()), dtype=torch.long)
    if self_loop and not homo:
        nun, _ = add_remaining_self_loops(edge_index=nun)
        nsn, _ = add_remaining_self_loops(edge_index=nsn)
        ntn, _ = add_remaining_self_loops(edge_index=ntn)
    data['node', 'u', 'node'].edge_index = nun
    data["node", 's', "node"].edge_index = nsn
    data["node", 't', "node"].edge_index = ntn

    if homo:
        data = data.to_homogeneous()

    if self_loop and homo:
        data.edge_index, _ = add_remaining_self_loops(data.edge_index)


    return [data]


def load_tfinance_or_tsocial(name, homo=1,self_loop=0):
    if name == "tfinance":
        datapath = "./datasets/tfinance"
        graph, label_dict = load_graphs(datapath)
        graph = graph[0]
        graph.ndata['label'] = graph.ndata['label'].argmax(1)

    elif name == "tsocial":
        datapath = "./datasets/tsocial"
        graph, label_dict = load_graphs(datapath)
        graph = graph[0]
    else:
        print("error dataset")

    edge_index = torch.vstack(graph.edges())

    if self_loop:
        edge_index, _ = add_remaining_self_loops(edge_index)

    node_feat = graph.ndata["feature"].float()
    label = graph.ndata['label']
    dataset = Data(x=node_feat, edge_index=edge_index, y=label)
    return [dataset]

