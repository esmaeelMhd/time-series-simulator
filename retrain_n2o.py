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
    parser.add_argument('--model', type=str, required=False, default='DLinear',
                        help='model name, options: [Autoformer, Informer, Transformer]')
    parser.add_argument('--lstm_type', type=str, required=False, default='lstm type: EncDec')

    # data loader
    parser.add_argument('--data_tag', type=str, default='New_CorrH', help='tag of the dataset')
    parser.add_argument('--dataset_name', type=str, default='wastewater.csv', help='dataset file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--setting', type=str, default='LSTM_New_CorrH_11F_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256Hidden_2LayerDim', help='saved model folder name')
    parser.add_argument('--data_start', type=str, default='2021-08-01 00:00:00+00:00', help='start point of the dataset')
    parser.add_argument('--data_end', type=str, default='2023-01-01 00:00:00+00:00', help='end point of the dataset')
    
    # general
    parser.add_argument('--seq_length', type=int, default=240, help='sequence length of the trained model')
    parser.add_argument('--val_ratio', type=float, default=0.05, help='validation ratio')
    parser.add_argument('--val_length', type=int, default=1440, help='validation episode length')
    parser.add_argument('--retrain_lr', type=float, default=1e-3, help='retrain learning rate')
    parser.add_argument('--retrain_wd', type=float, default=1e-9, help='retrain weight decay')
    parser.add_argument('--loss_function', type=str, default='mse', help='loss function: mse or dilate')
    parser.add_argument('--vall_loss_type', type=str, default='mse', help='validation loss function: mse, dtw, tdi')

    
    # experiments
    parser.add_argument('--experiment', type=int, default=1, help='type of the retrain experiment')
    parser.add_argument('--retrain_number', type=int, default=1, help='number of the retrain')
    parser.add_argument('--retrain_epochs', type=int, default=10, help='number of retrain epochs')
    parser.add_argument('--const_episode_length', type=int, default=60, help='constant episode length')
    parser.add_argument('--min_episode_length', type=int, default=10, help='minimum episode length')
    parser.add_argument('--max_episode_length', type=int, default=120, help='maximum episode length')
    parser.add_argument('--retrain_mode', type=str, default='batch_pred', help='type of the retrain: single_pred or batch_pred')
    parser.add_argument('--use_actual_action', type=bool, default=True, help='wether to use actual actions in retraining or not')
    parser.add_argument('--num_small_batches', type=int, default=20, help='number of consecutive small batches when the episode start is random')
    parser.add_argument('--gap_flag', type=str, default='no_gap', help='gap between episodes: 1- no_gap, 2- small_gap, 3- large_gap, 4- custom_gap')
    parser.add_argument('--custom_gap', type=int, default=10, help='custom gap value')
    parser.add_argument('--scheduled_sampling', type=bool, default=True, help='wether to use scheduled sampling in retraining or not')
    parser.add_argument('--ss_prob', type=float, default=0.5, help='the prob factor for scheduled sampling')
    parser.add_argument('--retrain_method', type=str, default='v1', help='type of the retrain: epochs')
    parser.add_argument('--sepp_gap', type=bool, default=False, help='wether to use gap in SEPP')
    parser.add_argument('--do_early_stop', type=bool, default=True, help='wether to use early stopping')
    parser.add_argument('--patience', type=int, default=3, help='patience of early stopping')
    parser.add_argument('--alpha_dilate', type=float, default=0.5, help='alpha for dilate loss')
    parser.add_argument('--gamma_dilate', type=float, default=0.001, help='gamma for dilate loss')
    parser.add_argument('--dilate_all', type=bool, default=True, help='wether to use dilate for all features')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # name of the retrain checkpoint
    parser.add_argument('--retrain_name', type=str, default='retrained_checkpoint', help='the retrain checkpoint name')
    
    # Parser
    retrain_args = parser.parse_args()
    
    retrain_args.retrain_method = 'v2'
    retrain_args.sepp_gap = True

    retrain_args.model = 'LSTM'
    retrain_args.lstm_type = ''
    retrain_args.data = 'DQAPTQN'
    retrain_args.dataset_name = 'DQAPTQN_Fredericia_2021_2022.csv'
    retrain_args.setting = 'N2O_LSTM_DQAPTQN_13F_1Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L'
    retrain_args.data_start = '2021-01-01 00:00:00+00:00'
    retrain_args.data_end = '2021-01-15 00:00:00+00:00'
    retrain_args.seq_length = 240
    retrain_args.val_ratio = 0.1
    retrain_args.val_length = 60
    retrain_args.retrain_lr = 1e-6
    retrain_args.retrain_wd = 1e-9
    retrain_args.loss_function = 'dilate'
    retrain_args.dilate_all = True
    retrain_args.alpha_dilate = 0.6
    retrain_args.gamma_dilate = 0.01
    retrain_args.val_loss_type = 'dtw'
    retrain_args.use_amp = False
    retrain_args.scheduled_sampling = False
    retrain_args.ss_prob = 0.5
    retrain_args.do_early_stop = True
    retrain_args.patience = 3
    
    # Experiment Setup
    '''
    Experiments:
        1- Constant episode start, and Constant episode length
        2- Constant episode start, and Random episode length
        3- Random episode start, and Constant episode length 
        4- Random episode start, and Random episode length
    '''
    retrain_args.experiment = 1
    retrain_args.retrain_number = 1
    
    retrain_args.retrain_epochs = 1
    # Mode options: single_pred, batch_pred
    retrain_args.retrain_mode = 'batch_pred'
    retrain_args.use_actual_actions = True
    
    # Gap
    # small_gap = episode_length, large_gap = seq_len + episode_length
    retrain_args.gap_flag = 'small_gap'
    retrain_args.custom_gap = 240 
    
    # Epsides
    retrain_args.const_episode_length = 60
    retrain_args.min_episode_length = 10
    retrain_args.max_episode_length = 480
    retrain_args.num_small_batches = 20
               
    return retrain_args

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

retrain_args = make_args()
df_raw = make_df_raw(retrain_args)
models_list = list([retrain_args.setting])

if __name__ == '__main__':
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
        