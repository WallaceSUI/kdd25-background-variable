import argparse

# arguments setting
def parse_args(): 
    parser = argparse.ArgumentParser(description='learning framework for RS')
    parser.add_argument('--dataset', type=str, default='coat', help='Choose from {yahooR3, coat, simulation}')

    parser.add_argument('--base_model_args', type=dict, default={'emb_dim': 4, 'learning_rate': 0.01, 'weight_decay': 5e-4}, 
                help='base model arguments.')
    
    parser.add_argument('--training_args', type=dict, default = {'batch_size': 128, 'batch_size_prop':1024, 'learning_rate_pretrain': 0.01, 'weight_decay_pretrain': 5e-4, 'learning_rate_prop': 0.01, 'weight_decay_prop': 5e-4}, 
                help='training arguments.')

    parser.add_argument('--seed', type=int, default=0, help='global general random seed.')

    parser.add_argument('--monte_sample_num', type=int, default=50, help='monte carlo smaple size')
    parser.add_argument('--sigma_y', type=float, default=2)
    parser.add_argument('--sigma_o', type=float, default=2)
    parser.add_argument('--optuna_continue', default=False, action='store_true')
    parser.add_argument('--optuna_name', type=str, default='main', help='exp_name for optuna')



    return parser.parse_args()