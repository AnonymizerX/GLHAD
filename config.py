GLOBAL_SEED = 42

large_graph = ["yelp", "amazon", "tfinance", "tsocial"]
homo_graph = ["tfinance", "tsocial"]


def default_config():
    config = {
        # Generally fixed
        'num_epochs': 200,
        # 'num_reduce_layers': 1,  # num of reduce heads
        'A_embed': False,        # whether use A embedding
        'out_norm': False,        # whether use normalization before prediction
        'out_mlp': True,        # whether use mlp for prediction head, otherwise linear
        'adamw': False,
        'train_val_test': [0.40, 0.20, 0.40],
        'homo': 1,
        'multi_layer_concate': False,
        'mode': 'Train',
        "filter_type":'dis',

        # params need to search
        'lr': 0.01,
        'wd': 5e-4,
        'dropout': 0.5,
        'beta': 0.,
        'K': 5,
        'hidden_channels': 32,
        # 'gamma': 0.,
        # 'method': 'norm2',
    }
    return config


def glhad(ds='amazon'):
    config = default_config()

    if ds in ("tsocial", "tfiance"):
        config['hidden_channels'] = 10
    return config
