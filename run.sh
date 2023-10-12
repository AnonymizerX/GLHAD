# YelpChi

# Hetero
# MF1: 80.95(0.23), AUC: 92.81(0.14), GMEAN: 80.56(1.08)
python train.py --wandb_mode disabled --dataset yelp --homo 0 \
--lr 0.005 --wd 0.001 --K 1 --hidden_channels 64 --dropout 0.1 --num_epochs 200  --gpu 0  --parameter_matrix 1 --multi_layer_concate 1 --filter_type dis --self_loop 1 --cpu -1 --num_train 5


# Homo
# MF1: 76.02(0.25), AUC: 88.38(0.18), GMEAN: 74.54(0.59)
python train.py --wandb_mode disabled --dataset yelp --homo 1 \
--lr 0.005 --wd 0.001 --K 1 --hidden_channels 64 --dropout 0.1 --num_epochs 200  --gpu 0  --parameter_matrix 1 --multi_layer_concate 1 --filter_type dis --self_loop 1 --cpu -1 --num_train 5



# Amazon
# Hetero
# MF1: 93.14(0.49), AUC: 97.59(0.13), GMEAN: 90.88(0.52)
python train.py --wandb_mode disabled --dataset amazon --homo 0 \
--lr 0.04 --wd 0.001 --K 1 --hidden_channels 32 --dropout 0.1 --num_epochs 200  --parameter_matrix 1 --multi_layer_concate 1 --filter_type mix --self_loop 1 --cpu -1  --gpu 1 --num_train 5


# Homo
# MF1: 92.99(0.19), AUC: 98.11(0.12), GMEAN: 91.34(0.09)
python train.py --wandb_mode disabled --dataset amazon --homo 1 \
--lr 0.04 --wd 0.001 --K 1 --hidden_channels 32 --dropout 0.1 --num_epochs 200 --parameter_matrix 1 --multi_layer_concate 1 --filter_type dis --self_loop 1 --cpu -1  --gpu 0 --num_train 5



# T-Finance
# MF1: 92.08(0.25), AUC: 97.00(0.27), GMEAN: 89.28(0.65)
python train.py --wandb_mode disabled --dataset tfinance --homo 1 \
--lr 0.1 --wd 0 --K 4 --hidden_channels 16 --dropout 0.4 --num_epochs 200  --gpu 0  --cpu -1 --multi_layer_concate 1 --num_train 5


# T-Social
# MF1: 94.10(0.35), AUC: 98.95(0.10), GMEAN: 92.65(0.37) 
python train.py --wandb_mode disabled --dataset tsocial --homo 1 \
--gpu -1  --lr 0.1 --wd 0 --K 3 --hidden_channels 10 --dropout 0.0 --num_epochs 100 --run_id 1 --parameter_matrix 1 --multi_layer_concate 1 --filter_type dis --self_loop 0 --cpu -1 --num_train 5