import math
import os
import torch
import random
import torch_geometric

import os.path as osp
import numpy as np
import scipy.sparse as sp


from torch_geometric.utils import degree
from torch_sparse import SparseTensor

from torch_geometric.utils import (add_remaining_self_loops,
                                   remove_self_loops)

from load_large_graph.dataset import load_Amazon_Dataset, load_Yelp_Dataset, load_tfinance_or_tsocial
import config as c


def get_dataset(name: str, homo, self_loop):
    name = name.lower()
    if name in ("yelp"):
        dataset = load_Yelp_Dataset(homo, self_loop)
    elif name in ("amazon"):
        dataset = load_Amazon_Dataset(homo, self_loop)
    elif name in ("tfinance", "tsocial"):
        dataset = load_tfinance_or_tsocial(name, homo=1, self_loop=self_loop)
    else:
        raise NameError('dataset not found, name error.')
    return dataset


def dataset_split(
        y,
        num_nodes,
        num_classes,
        train_r=0.48,
        val_r=0.32,
        test_r=0.20, ds='yelp'):

    # assert train_r + val_r + test_r == 1

    from sklearn.model_selection import train_test_split

    index = list(range(num_nodes))
    if ds == 'amazon':
        index = list(range(3305, num_nodes))
    labels_cpu = y.cpu()

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels_cpu[index], stratify=labels_cpu[index],
                                                            train_size=train_r,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)

    train_mask = torch.zeros([num_nodes]).bool()
    val_mask = torch.zeros([num_nodes]).bool()
    test_mask = torch.zeros([num_nodes]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1

    return train_mask, val_mask, test_mask


def get_dataset_split(N, ds, data, config):

    y = data.y if config['homo'] else data['node'].y

    num_classes = y.unique().shape[0]
    train_mask, val_mask, test_mask = \
        dataset_split(
            y, N, num_classes, *config['train_val_test'], ds=ds)

    return train_mask, val_mask, test_mask


def scipy_coo_matrix_to_torch_sparse_tensor(sparse_mx):
    indices1 = torch.from_numpy(
        np.stack([sparse_mx.row, sparse_mx.col]).astype(np.int64))
    values1 = torch.from_numpy(sparse_mx.data)
    shape1 = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices=indices1, values=values1, size=shape1)


def cal_filter(edge_index, num_nodes):
    edge_index = edge_index.cpu()
    N = num_nodes

    # A
    edge_index, _ = remove_self_loops(edge_index=edge_index)
    edge_index_sl, _ = add_remaining_self_loops(edge_index=edge_index)

    # D
    adj_data = np.ones([edge_index.shape[1]], dtype=np.float32)
    adj_sp = sp.csr_matrix(
        (adj_data, (edge_index[0], edge_index[1])), shape=[N, N])

    adj_sl_data = np.ones([edge_index_sl.shape[1]], dtype=np.float32)
    adj_sl_sp = sp.csr_matrix(
        (adj_sl_data, (edge_index_sl[0], edge_index_sl[1])), shape=[N, N])

    # D-1/2
    deg = np.array(adj_sl_sp.sum(axis=1)).flatten()
    deg_sqrt_inv = np.power(deg, -0.5)
    deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0.0
    deg_sqrt_inv = sp.diags(deg_sqrt_inv)

    # filters
    DAD = sp.coo_matrix(deg_sqrt_inv * adj_sp * deg_sqrt_inv)
    DAD = scipy_coo_matrix_to_torch_sparse_tensor(DAD)

    return DAD


def cal_DAD(edge_index, num_nodes, dataset, edge_name, add_self_loop=0):

    if os.path.exists(osp.join('datasets/processed', dataset, "".join(edge_name), 'DAD.pt')):
        DAD = torch.load(osp.join('datasets/processed',
                         dataset, "".join(edge_name), f'DAD_{add_self_loop}.pt'))
        return DAD

    edge_index = edge_index.cpu()
    N = num_nodes

    # A
    edge_index, _ = remove_self_loops(edge_index=edge_index)
    # self loop

    adj_data = np.ones([edge_index.shape[1]], dtype=np.float32)
    adj_sp = sp.csr_matrix(
        (adj_data, (edge_index[0], edge_index[1])), shape=[N, N])
    # adj_sp = sp.csr_matrix((adj_data, (edge_index[0], edge_index[1])), shape=[N, N])

    # D
    deg = np.array(adj_sp.sum(axis=1)).flatten().clip(min=1)
    # D-1/2
    deg_sqrt_inv = np.power(deg, -0.5)
    deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0.0
    deg_sqrt_inv = sp.diags(deg_sqrt_inv)

    # filters
    DAD = sp.coo_matrix(deg_sqrt_inv * adj_sp * deg_sqrt_inv)
    DAD = scipy_coo_matrix_to_torch_sparse_tensor(DAD)

    # os.mkdir(osp.join('datasets/processed', dataset, edge_name))
    os.makedirs(osp.join('datasets', 'processed', dataset,
                "".join(edge_name)), exist_ok=True)
    torch.save(DAD, osp.join('datasets/processed',
               dataset, "".join(edge_name), f'DAD_{add_self_loop}.pt'))

    return DAD


def cal_low_high_filter(dev, N, beta, dataset, edge_name, edge_index, filter_name, add_self_loop=0):

    if os.path.exists(osp.join('datasets/processed', dataset, filter_name, "".join(edge_name), f'filter_l_{add_self_loop}.pt')):
        filter_l = torch.load(
            osp.join('datasets/processed', dataset, filter_name, "".join(edge_name), f'filter_l_{add_self_loop}.pt'))
        filter_h = torch.load(
            osp.join('datasets/processed', dataset, filter_name, "".join(edge_name), f'filter_h_{add_self_loop}.pt'))
        return filter_l.to(dev), filter_h.to(dev)

    DAD = cal_DAD(edge_index, N, dataset=dataset,
                  edge_name=edge_name, add_self_loop=add_self_loop)

    I_indices = torch.LongTensor([[i, i] for i in range(N)]).t()
    I_values = torch.tensor([1. for _ in range(N)])
    I_sparse = torch.sparse.FloatTensor(
        indices=I_indices, values=I_values, size=torch.Size([N, N]))

    if filter_name == 'l_sym':
        filter_l = SparseTensor.from_torch_sparse_coo_tensor(
            beta * I_sparse + DAD)
        filter_h = SparseTensor.from_torch_sparse_coo_tensor(
            (1. - beta) * I_sparse - DAD)
    elif filter_name == 'half_l_sym':
        filter_l = SparseTensor.from_torch_sparse_coo_tensor(
            beta * I_sparse + DAD*0.5)
        filter_h = SparseTensor.from_torch_sparse_coo_tensor(
            (1. - beta) * I_sparse - DAD*0.5)
    else:
        print("error")
        exit()

    os.makedirs(osp.join('datasets', 'processed', dataset,
                filter_name, "".join(edge_name)), exist_ok=True)
    torch.save(filter_l, osp.join('datasets/processed',
               dataset, filter_name, "".join(edge_name), f'filter_l_{add_self_loop}.pt'))
    torch.save(filter_h, osp.join('datasets/processed',
               dataset, filter_name, "".join(edge_name), f'filter_h_{add_self_loop}.pt'))
    return filter_l.to(dev), filter_h.to(dev)


def set_seed(seed=28):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # https://pytorch.org/docs/stable/notes/randomness.html
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = str(':4096:8')
    # pyg
    torch_geometric.seed_everything(seed)


def cal_edge_type(train_mask, edge_index, y):

    mask_src = train_mask[edge_index[0].cpu()]
    mask_trg = train_mask[edge_index[1].cpu()]

    known_edge_mask = torch.logical_and(mask_src, mask_trg)

    src_label = y[edge_index[0].cpu()]
    trg_label = y[edge_index[1].cpu()]

    edge_type = ~torch.logical_xor(src_label, trg_label)

    return known_edge_mask, edge_type.float()


def cal_all_edge_type(train_mask, edge_index, y):
    if isinstance(edge_index, dict):
        known_edge_mask = {}
        edge_type = {}
        for curr_edge_name, curr_edge_index in edge_index.items():

            known_edge_mask[curr_edge_name], edge_type[curr_edge_name] = cal_edge_type(
                train_mask, curr_edge_index, y)
    else:
        known_edge_mask, edge_type = cal_edge_type(train_mask, edge_index, y)

    return known_edge_mask, edge_type


def cal_node_homo(edge_index, labels, N, dev):
    src_index, trg_index = edge_index
    homo_edge = ~torch.logical_xor(
        labels[src_index].cpu(), labels[trg_index].cpu())
    local_homo = torch.zeros(N)
    local_homo = local_homo.scatter_add(
        dim=0, index=src_index.cpu(), src=homo_edge.float())
    deg = degree(src_index.cpu(), num_nodes=N)
    local_homo = local_homo/deg
    # inf
    local_homo[deg == 0] = 1.0
    local_homo = local_homo.unsqueeze(1)
    if torch.isnan(local_homo).sum() or torch.isinf(local_homo).sum():
        print("error")
    return local_homo.to(dev)


def node_homo(edge_index, labels, N):
    src_index, trg_index = edge_index
    homo_edge = ~torch.logical_xor(
        labels[src_index].cpu(), labels[trg_index].cpu())
    local_homo = torch.zeros(N)
    local_homo = local_homo.scatter_add(
        dim=0, index=src_index.cpu(), src=homo_edge.float())
    deg = degree(src_index.cpu(), num_nodes=N)
    local_homo = local_homo/deg
    # inf
    local_homo[deg == 0] = 1.0
    local_homo = local_homo.unsqueeze(1)
    if torch.isnan(local_homo).sum() or torch.isinf(local_homo).sum():
        print("error")
    return local_homo


def cal_local_homo_ratio(edge_index, labels, N, dev):
    if isinstance(edge_index, dict):
        local_homo = {}
        for curr_edge_name, curr_edge_index in edge_index.items():
            local_homo[curr_edge_name] = cal_node_homo(
                curr_edge_index, labels, N, dev)
    else:
        local_homo = cal_node_homo(edge_index, labels, N, dev)

    return local_homo
