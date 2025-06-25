import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR) 

import numpy as np
import pandas as pd
import pickle
import warnings
import datetime
import os

from data_stamp import DataStamp

import torch

import random
import argparse

from Retrainer import Retrainer

#%% Plot and device options

try:
    plt.rcParams['font.family'] = 'Times New Roman'
except Exception as e:
    pass

plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.xmargin'] = 0.02
plt.rcParams['axes.ymargin'] = 0.04
fig_dpi = 500
# plt.switch_backend('qt5agg')

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
# print(f"{device}" " is available.")

# Frequency of the dataset
freq = datetime.timedelta(minutes=1)

# np.random.seed(2000)

#%%

def make_args():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Retrain Models for DRL Environment')

    # basic config
    parser.add_argument('--model', type=str, required=False, default=None,
                        help='model name, options: [Autoformer, Informer, Transformer]')
    parser.add_argument('--lstm_type', type=str, required=False, default=None)

    # data loader
    parser.add_argument('--data_tag', type=str, default=None, help='tag of the dataset')
    parser.add_argument('--dataset_name', type=str, default=None, help='dataset file')
    parser.add_argument('--checkpoints', type=str, default=None, help='location of model checkpoints')
    parser.add_argument('--setting', type=str, default=None, help='saved model folder name')
    parser.add_argument('--data_start', type=str, default=None, help='start point of the dataset')
    parser.add_argument('--data_end', type=str, default=None, help='end point of the dataset')
    
    # general
    parser.add_argument('--seq_length', type=int, default=None, help='sequence length of the trained model')
    parser.add_argument('--val_ratio', type=float, default=None, help='validation ratio')
    parser.add_argument('--val_length', type=int, default=None, help='validation episode length')
    parser.add_argument('--retrain_lr', type=float, default=None, help='retrain learning rate')
    parser.add_argument('--retrain_wd', type=float, default=None, help='retrain weight decay')
    parser.add_argument('--loss_function', type=str, default=None, help='loss function: mse or dilate')
    parser.add_argument('--vall_loss_type', type=str, default=None, help='validation loss function: mse, dtw, tdi')

    
    # experiments
    parser.add_argument('--experiment', type=int, default=None, help='type of the retrain experiment')
    parser.add_argument('--retrain_number', type=int, default=None, help='number of the retrain')
    parser.add_argument('--retrain_epochs', type=int, default=None, help='number of retrain epochs')
    parser.add_argument('--const_episode_length', type=int, default=None, help='constant episode length')
    parser.add_argument('--min_episode_length', type=int, default=None, help='minimum episode length')
    parser.add_argument('--max_episode_length', type=int, default=None, help='maximum episode length')
    parser.add_argument('--retrain_mode', type=str, default=None, help='type of the retrain: single_pred or batch_pred')
    parser.add_argument('--use_actual_action', type=bool, default=None, help='wether to use actual actions in retraining or not')
    parser.add_argument('--num_small_batches', type=int, default=None, help='number of consecutive small batches when the episode start is random')
    parser.add_argument('--gap_flag', type=str, default=None, help='gap between episodes: 1- no_gap, 2- small_gap, 3- large_gap, 4- custom_gap')
    parser.add_argument('--custom_gap', type=int, default=None, help='custom gap value')
    parser.add_argument('--scheduled_sampling', type=bool, default=None, help='wether to use scheduled sampling in retraining or not')
    parser.add_argument('--ss_prob', type=float, default=None, help='the prob factor for scheduled sampling')
    parser.add_argument('--retrain_method', type=str, default=None, help='type of the retrain: epochs')
    parser.add_argument('--sepp_gap', type=bool, default=None, help='wether to use gap in SEPP')
    parser.add_argument('--do_early_stop', type=bool, default=None, help='wether to use early stopping')
    parser.add_argument('--patience', type=int, default=None, help='patience of early stopping')
    parser.add_argument('--alpha_dilate', type=float, default=None, help='alpha for dilate loss')
    parser.add_argument('--gamma_dilate', type=float, default=None, help='gamma for dilate loss')
    parser.add_argument('--dilate_all', type=bool, default=None, help='wether to use dilate for all features')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=None)
    
    # name of the retrain checkpoint
    parser.add_argument('--retrain_name', type=str, default=None, help='the retrain checkpoint name')
               
    return parser.parse_args()

#%% 
def make_df_raw(retrain_args):
    home_dir = os.path.expanduser("~")
    if not os.path.exists(home_dir + '/raid/vz75cp/'):
        dataset_dir = './datasets/'
    else:
        print('Loading from raid ...')
        dataset_dir = os.path.join(home_dir, 'raid/vz75cp/datasets/')
        
    df_raw = pd.read_csv(dataset_dir + retrain_args.dataset_name)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.set_index(["date"])
    if not df_raw.index.is_monotonic:
        df_raw = df_raw.sort_index()
    
    data_start = pd.to_datetime(retrain_args.data_start)
    data_end = pd.to_datetime(retrain_args.data_end)
    df_raw = df_raw.loc[data_start:data_end]
    
    return df_raw
    
#%%

def main():   
    # Make retrain args
    retrain_args = make_args()
   
    # Set arguments
    '''
    Experiments:
        1- Constant episode start, and Constant episode length
        2- Constant episode start, and Random episode length
        3- Random episode start, and Constant episode length 
        4- Random episode start, and Random episode length
    '''
    
    defaults = {
        'retrain_method': 'v2',
        'sepp_gap': True,
        'model': 'LSTM',
        'lstm_type': '',
        'data': 'IOTQCfP',
        'dataset_name': 'wastewater.csv',
        'setting': 'LSTM_IOTQCfP_12F_1Out_timeF_Scaled_240Seq_10Pred_0Label_128Batch_1e-06LR_80H_1L',
        'data_start': '2021-08-15 00:00:00+00:00',
        'data_end': '2021-08-20 00:00:00+00:00',
        'seq_length': 240,
        'val_ratio': 0.05,
        'val_length': 60,
        'retrain_lr': 1e-6,
        'retrain_wd': 1e-9,
        'loss_function': 'dilate',
        'dilate_all': False,
        'alpha_dilate': 0.6,
        'gamma_dilate': 0.01,
        'val_loss_type': 'dtw',
        'use_amp': False,
        'scheduled_sampling': False,
        'ss_prob': 0.5,
        'do_early_stop': True,
        'patience': 3,
        
        # Experiment Setup
        'experiment': 1,
        'retrain_number': 1,
        'retrain_epochs': 20,
        
        # Mode options: single_pred, batch_pred
        'retrain_mode': 'batch_pred',
        'use_actual_actions': True,
        
        # Gap
        # small_gap = episode_length, large_gap = seq_len + episode_length
        'gap_flag': 'small_gap',
        'custom_gap': 240,
        
        # Epsiodes
        'const_episode_length': 240,
        'min_episode_length': 10,
        'max_episode_length': 480,
        'num_small_batches': 20
        }

    for arg, value in defaults.items():
        if getattr(retrain_args, arg, None) is None:
            setattr(retrain_args, arg, value)
    
    # Start retrain process
    df_raw = make_df_raw(retrain_args)
    models_list = list([retrain_args.setting])
    
    for model in models_list:
        # Empty the cache
        torch.cuda.empty_cache()
        
        # Load the args file of the model
        ARGS_PATH = './args/' + model + '/'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with open(ARGS_PATH + 'args.pkl', 'rb') as file:
                args = pickle.load(file)
        
        # Create the stamp dataset for LTSF models
        if args.model != 'LSTM':
            data_stamp = DataStamp(df_raw.rename_axis('date').reset_index(level=0), args)
            df_stamp = data_stamp.create_data_stamps() 
        
        # Create an instance of the retrainer and start retraining
        
        
        retrainer = Retrainer(args, df_raw, retrain_args, device)
        if retrain_args.retrain_method == 'sepp':
            retrainer.start_retrain_sepp()
        else:
            retrainer.start_retrain()
        
        train_ys = retrainer.opt.train_ys
        # train_dictionary = retrainer.train_dictionary
        # results_dictionary = retrainer.results_dictionary
        
        retrainer.test_retrain()
        

if __name__ == '__main__':
    main()