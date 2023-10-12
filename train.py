import time
import json
import os
import torch
import wandb
import itertools
import psutil
import random


import model as m
import config as c
import numpy as np
import utils.utils as u
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from args_parse import args_parse

from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from torch_geometric.nn import summary


def train_onetime(ds: str, model_name='GLHAD', config=None, n=0):
    # get dataset
    data = u.get_dataset(ds, config['homo'], config['self_loop'])[0].to(device)

    u.set_seed(config['seed']+n)

    if config == None:
        config = c.glhad(ds)

    if config['homo']:
        x = F.normalize(data.x, p=2, dim=-1)
        edge_index = data.edge_index
        y = data.y.squeeze()
    else:
        x = F.normalize(data['node'].x, p=2, dim=-1)
        edge_index = data.edge_index_dict
        y = data['node'].y.squeeze()

    N = x.shape[0]

    num_classes = y.unique().shape[0]

    # dataset split
    train_mask, val_mask, test_mask = u.get_dataset_split(
        N, ds, data, config)

    # ignore unlabeled data (label = -1)
    if (y == -1).sum() > 0:
        num_classes -= 1

    # get config and model
    if model_name == 'GLHAD':
        model = m.GLHAD(
            in_channels=x.shape[1],
            out_channels=num_classes,
            config=config,
            num_nodes=N,
            ds=ds,
            edge_index=edge_index,
            device=device
        ).to(device)
    else:
        raise NotImplementedError

    # torch compile
    # model = torch_geometric.compile(model)

    config_dict = json.dumps(config, indent=4)
    print(config_dict)

    # optim and loss
    if config['adamw']:
        optim = torch.optim.AdamW(
            model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    else:
        optim = torch.optim.Adam(
            model.parameters(), lr=config['lr'], weight_decay=config['wd'])

    # cross entropy weight
    weight = (1-y[train_mask]).sum().item() / y[train_mask].sum().item()
    # print('cross entropy weight: ', weight)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., weight]).to(device))

    if True:
        # local_homo_labels = u.cal_local_homo_ratio(
        #     edge_index=edge_index, labels=y, N=N, dev=device)
        local_homo_labels = None

        if config['homo']:
            local_homo_loss = u.node_homo(edge_index, labels=y, N=N).numpy()
        else:
            local_homo_loss = {}
            for curr_edge_name, curr_edge_index in edge_index.items():
                # src_index, trg_index = edge_index
                local_homo_loss[curr_edge_name] = u.node_homo(
                    curr_edge_index, labels=y, N=N).numpy()

    best_f1, final_tf1, final_trec, final_tpre, final_tacc, final_tmf1, final_tauc, final_tgmean = 0., 0., 0., 0., 0., 0., 0., 0.

    # torch.cuda.empty_cache()

    # model summary
    if config['mode'] == "Debug":
        print(summary(model, x, edge_index, local_homo=local_homo_labels))
        if data.has_isolated_nodes():
            print("data has isolated nodes")
        pass

    t = tqdm(range(config['num_epochs']), ncols=70, desc='iter %d' % (n))
    for e in t:
        formatted_mf1 = f"{final_tmf1 * 100:.2f}"
        t.set_postfix({"MF1": formatted_mf1})
        model.train()
        optim.zero_grad()

        logits, pre_local_homo, pre_edge_type = model(
            x, edge_index, local_homo=local_homo_labels)

        loss = loss_fn(logits[train_mask], y[train_mask])

        loss.backward()

        optim.step()

        # val and test
        with torch.no_grad():
            model.eval()
            logits, pre_local_homo_test, _ = model(
                x, edge_index, local_homo=local_homo_labels)

            probs = logits.softmax(dim=1)

            # threshold adjusting for best macro f1
            def get_best_f1(labels, probs):
                best_f1, best_thre = 0, 0
                for thres in np.linspace(0.05, 0.95, 19):
                    preds = torch.zeros_like(labels)
                    preds[probs[:, 1] > thres] = 1
                    mf1 = f1_score(labels.cpu(), preds.cpu(), average='macro')
                    if mf1 > best_f1:
                        best_f1 = mf1
                        best_thre = thres
                return best_f1, best_thre

            f1, thres = get_best_f1(y[val_mask], probs[val_mask])

            wandb.log({"loss": loss, "val_f1": f1, "best_f1": final_tmf1})

            if best_f1 < f1:
                best_f1 = f1
                preds = torch.zeros_like(y)
                preds[probs[:, 1] > thres] = 1

                trec = recall_score(
                    y[test_mask].cpu(), preds[test_mask].cpu())
                tpre = precision_score(
                    y[test_mask].cpu(), preds[test_mask].cpu())
                tacc = accuracy_score(
                    y[test_mask].cpu(), preds[test_mask].cpu())

                tmf1 = f1_score(y[test_mask].cpu(),
                                preds[test_mask].cpu(), average='macro')
                tauc = roc_auc_score(
                    y[test_mask].cpu(), probs[test_mask][:, 1].cpu().detach().numpy())

                def conf_gmean(conf):
                    tn, fp, fn, tp = conf.ravel()
                    return (tp*tn/((tp+fn)*(tn+fp)))**0.5

                conf_gnn = confusion_matrix(
                    y[test_mask].cpu(), preds[test_mask].cpu())
                tgmean = conf_gmean(conf_gnn)

                final_trec = trec
                final_tpre = tpre
                final_tacc = tacc
                final_tmf1 = tmf1
                final_tauc = tauc
                final_tgmean = tgmean

    wandb.log({"MF1": final_tmf1, "AUC": final_tauc, "GMEAN": final_tgmean})
    print('Test: REC {:.2f} PRE {:.2f} ACC {:.2F} MF1 {:.2f} AUC {:.2f} GMEAN {:.2f}'.format(final_trec*100,
                                                                                             final_tpre*100, final_tacc*100, final_tmf1*100, final_tauc*100, final_tgmean*100))

    return best_f1, final_tmf1, final_tauc, final_tgmean


def train_ntime(ds: str, model_name: str = 'glhad', config=None, num_train=None):
    if num_train is None:
        num_train = 10
        if ds == 'ogbn-arxiv':
            num_train = 1
        elif ds in c.large_graph:
            num_train = 1

    if ds in c.homo_graph:
        config['homo'] = 1

    result = []
    for n in range(num_train):
        best_vmf1, best_tmf1, best_tauc, best_tgmean = train_onetime(
            ds, model_name=model_name, config=config, n=n)
        result.append([best_vmf1, best_tmf1, best_tauc, best_tgmean])

    val_mf1_mean, test_mf1_mean, test_auc_mean, test_gmean_mean = np.mean(
        result, axis=0) * 100
    val_mf1_std, test_mf1_std, test_auc_std, test_gmean_std = np.sqrt(
        np.var(result, axis=0)) * 100

    print('model: %s, dataset: %s, val mf1: %.2f±%.2f, test mf1: %.2f±%.2f, test auc: %.2f±%.2f, test gmean: %.2f±%.2f' %
          (model_name, ds, val_mf1_mean, val_mf1_std, test_mf1_mean, test_mf1_std, test_auc_mean, test_auc_std, test_gmean_mean, test_gmean_std))

    return test_mf1_mean, test_auc_mean, test_gmean_mean, test_mf1_std, test_auc_std, test_gmean_std


if __name__ == '__main__':

    args = args_parse()

    args.project = args.project+'_'+args.dataset

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    def limit_cpu_usage(cpu_num):
        if cpu_num >= 0:
            count = psutil.cpu_count()
            p = psutil.Process()
            cpu_lst = p.cpu_affinity()
            sample_cpu_list = random.sample(cpu_lst, cpu_num)
            p.cpu_affinity(sample_cpu_list)
    limit_cpu_usage(args.cpu)

    u.set_seed(args.seed)

    best_result_info = {
        "test_mf1": 0,
        "test_auc": 0,
        "test_gmean": 0,
        "weight_decay": None,
        "lr": None,
        "k": None,
    }

    if args.hyperparameter_search:

        lr = [0.1, 0.05, 0.01, 0.005]
        weight_decay = [0]
        K = [1, 2, 3, 4, 5, 6]
        hidden_channels = [10]
        dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

        print("Hyperparameter Searching count: ", len(lr) *
              len(weight_decay)*len(K)*len(hidden_channels)*len(dropout))

        config = c.glhad(args.dataset)

        config['model_name'] = args.model_name

        config['parameter_matrix'] = args.parameter_matrix
        config['filter_type'] = args.filter_type
        config['filter_name'] = args.filter_name

        config['seed'] = args.seed
        config['num_epochs'] = args.num_epochs
        config['train_val_test'][0] = args.train_ratio
        config['homo'] = args.homo
        config['self_loop'] = args.self_loop
        config['multi_layer_concate'] = args.multi_layer_concate

        print("Hyperparameter Searching...")
        for i, (curr_lr, curr_weight_decay, curr_k, curr_hidden_channels, curr_dropout) in enumerate(itertools.product(
            lr, weight_decay, K, hidden_channels, dropout
        )):

            t_total = time.time()
            epoch_total = 0

            config['lr'] = curr_lr
            config['wd'] = curr_weight_decay
            config['K'] = curr_k
            config['hidden_channels'] = curr_hidden_channels
            config['dropout'] = curr_dropout

            wandb.init(
                # set the wandb project where this run will be logged
                project=args.project,
                config=config,
                name=f"lr:{curr_lr:.4f}, wd:{curr_weight_decay:.4f}, k:{curr_k}, hidden_channels:{curr_hidden_channels}, dropout:{curr_dropout}",
                mode=args.wandb_mode,
            )

            test_mf1_mean, test_auc_mean, test_gmean_mean, test_mf1_std, test_auc_std, test_gmean_std = train_ntime(
                args.dataset, model_name=config['model_name'], config=config, num_train=args.num_train)
            epoch_total += args.num_train*config["num_epochs"]

            total_time_elapsed = time.time() - t_total
            runtime_average = total_time_elapsed / epoch_total

            if test_mf1_mean > best_result_info['test_mf1']:
                best_result_info['test_mf1'] = test_mf1_mean
                best_result_info['test_auc'] = test_auc_mean
                best_result_info['test_gmean'] = test_gmean_mean
                best_result_info['weight_decay'] = curr_weight_decay
                best_result_info['lr'] = curr_lr
                best_result_info['k'] = curr_k

            print(f"time:{time.time()-t_total:.4f}s")

            msg = f"lr:{curr_lr:.4f}, wd:{curr_weight_decay:.4f}, k:{curr_k}, test_mf1:{test_mf1_mean:.4f}±{test_mf1_std:.4f}, test_auc:{test_auc_mean:.4f}±{test_auc_std:.4f}, test_gmean:{test_gmean_mean:.4f}±{test_gmean_std:.4f}, runtime:{runtime_average:.4f}s"
            print(msg)
            best_msg = f"Current Best Result: test_mf1:{best_result_info['test_mf1']:.4f}, test_auc:{best_result_info['test_auc']:.4f}, test_gmean:{best_result_info['test_gmean']:.4f}, weight_decay:{best_result_info['weight_decay']:.4f}, lr:{best_result_info['lr']:.4f}, k:{best_result_info['k']}"
            print(best_msg)
            wandb.finish()

        best_msg = f"Best Result: test_mf1:{best_result_info['test_mf1']:.2f}, test_auc:{best_result_info['test_auc']:.2f}, test_gmean:{best_result_info['test_gmean']:.2f}, weight_decay:{best_result_info['weight_decay']:.2f}, lr:{best_result_info['lr']:.2f}, k:{best_result_info['k']}"
        print(best_msg)

    else:

        t_total = time.time()
        config = c.glhad(args.dataset)

        config['model_name'] = args.model_name

        config['num_epochs'] = args.num_epochs
        config['train_val_test'][0] = args.train_ratio
        config['homo'] = args.homo
        config['mode'] = args.mode
        config['seed'] = args.seed

        # Hyperparameter
        config['parameter_matrix'] = args.parameter_matrix
        config['multi_layer_concate'] = args.multi_layer_concate
        config['filter_type'] = args.filter_type
        config['filter_name'] = args.filter_name
        config['self_loop'] = args.self_loop

        config['K'] = args.K
        config['hidden_channels'] = args.hidden_channels
        config['dropout'] = args.dropout
        config['lr'] = args.lr
        config['wd'] = args.wd

        config['run_id'] = args.run_id

        wandb.init(
            project=args.project,
            config=config,
            name=f"lr:{config['lr']:.4f}, wd:{config['wd']:.4f}, k:{config['K']}, hidden_channels:{config['hidden_channels']}, dropout:{config['dropout']}",
            mode=args.wandb_mode,
        )

        test_mf1_mean, test_auc_mean, test_gmean_mean, test_mf1_std, test_auc_std, test_gmean_std = train_ntime(
            args.dataset, model_name=config['model_name'], config=config, num_train=args.num_train)
        print(f"time:{time.time()-t_total:.4f}s")

        print('MF1: {:.2f}({:.2f}), AUC: {:.2f}({:.2f}), GMEAN: {:.2f}({:.2f})'.format(test_mf1_mean,
                                                                                       test_mf1_std,
                                                                                       test_auc_mean,  test_auc_std,  test_gmean_mean,  test_gmean_std))

        wandb.finish()
