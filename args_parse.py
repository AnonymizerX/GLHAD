
import argparse

ds_list = [
    "yelp", "amazon", "tfinance", "tsocial"
]


def args_parse():
    parser = argparse.ArgumentParser(description='GAD')
    parser.add_argument("--project", default='exp1', type=str, help='exp name')
    parser.add_argument("--dataset", default='amazon',
                        dest='dataset', type=str, choices=ds_list, help='Dataset')
    parser.add_argument("--hyperparameter_search", default=False,
                        type=bool, help='Hyperparameter Searching')
    parser.add_argument("--num_train", default=1, type=int, help='num_train')

    parser.add_argument("--train_ratio", default=0.4,
                        type=float, help='defalut train ratio')
    parser.add_argument("--gpu", default=3, type=int, help='gpu id')
    parser.add_argument("--cpu", default=-1, type=int, help='cpu process num')

    parser.add_argument("--num_epochs", default=200,
                        type=int, help='num_epochs')
    parser.add_argument("--mode", default='Debug',
                        type=str, help='Debug or Release')
    parser.add_argument("--seed", default=2023, type=int, help='random seed')

    parser.add_argument("--homo", default=1, type=int,
                        help='homo graph or hetero graph')
    parser.add_argument("--self_loop", default=0,
                        type=int, help='add self loop')

    parser.add_argument("--model_name", default="GLHAD", type=str, choices=["GLHAD"],
                        help='model name')

    # model hyperparameters
    parser.add_argument("--multi_layer_concate", default=0, type=int,
                        help='concate multi layer')

    parser.add_argument("--parameter_matrix", default=1, type=int,
                        help='using the parameter matrix')
    parser.add_argument("--filter_type", default='dis', type=str, choices=['dis', 'mix'],
                        help='Separate filters, mixed filters')

    parser.add_argument("--filter_name", default='l_sym', type=str, choices=['l_sym', 'half_l_sym'],
                        help='filter group name')

    parser.add_argument("--lr", default=0.01, type=float, help='learning rate')
    parser.add_argument("--wd", default=5e-4, type=float, help='weight decay')
    parser.add_argument("--dropout", default=0.5, type=float, help='dropout')
    parser.add_argument("--K", default=4, type=int, help='model layer number')
    parser.add_argument("--hidden_channels", default=32,
                        type=int, help='hidden_channels')

    # debug
    parser.add_argument("--run_id", default=1, type=int, help='run_id')
    parser.add_argument("--wandb_mode", default="disabled", choices=['online', 'offline', 'disabled'],
                        type=str, help='wandb')

    args = parser.parse_args()
    return args
