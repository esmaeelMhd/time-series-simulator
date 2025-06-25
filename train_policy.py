"""
Created on April 2023
@author: Esmaeel Mohammadi

# =============================================================================
# This script is used to train and test different models
    1. Preprocessing of the data
    2. Convert data to Tensors for the model
    3. Convert Tensors to DataLoaders for the model
    4. Train the model
    5. Test the model
# =============================================================================
"""

import argparse
import os
import torch

import pickle

import pandas as pd

from exp.exp_main import Exp_Main
from exp.exp_lstm import ExpLSTM

import random
import numpy as np

#%%

def make_args():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model', type=str, required=False, default='DLinear',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
    # Formers 
    parser.add_argument('--embed_type', type=int, default=1, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    
    # Added by Me
    parser.add_argument('--scale', type=bool, default=True, help='whether to scale the data or not')
    parser.add_argument('--simulator', type=bool, default=True, help='whether to use as a simulator or not')
    parser.add_argument('--setting', type=str, default='LSTM_Phosphorous', help='args.setting file')
    parser.add_argument('--do_early_stop', type=bool, default=True, help='whether to early stop or not')
    parser.add_argument('--do_lr_opt', type=bool, default=True, help='whether to otimize learning rate or not')
    parser.add_argument('--time_scaled', type=str, default='Unscaled', help='Whether the time features scaled or not')
    parser.add_argument('--hidden_dim', type=int, default=512, help='dimension of model for LSTM')
    parser.add_argument('--layer_dim', type=int, default=2, help='num of LSTM layers')
    parser.add_argument('--in_features', type=int, default=10, help='LSTM model input size')
    parser.add_argument('--out_features', type=int, default=10, help='LSTM model output size')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay for LSTM')
    parser.add_argument('--data_tag', type=str, required=False, default='CorrH', help='dataset tag')
    parser.add_argument('--self_supervised', type=bool, default=False, help='whether to use self supervised training')
    parser.add_argument('--random_episode_length', type=bool, default=False, help='whether to use random episode length')
    parser.add_argument('--min_episode_length', type=int, default=10, help='minimum episode length')
    parser.add_argument('--max_episode_length', type=int, default=120, help='maximum episode length')
    parser.add_argument('--alpha_dilate', type=float, default=0.5, help='alpha for dilate loss')
    parser.add_argument('--gamma_dilate', type=float, default=0.001, help='gamma for dilate loss')
    parser.add_argument('--dilate_all', type=bool, default=True, help='wether to use dilate for all features')
    parser.add_argument('--ctrl_vars', type=str, default='IN_METAL_Q', help='control variable in the dataset like Metal amount')
    parser.add_argument('--ind_vars', type=str, default='TEMPERATURE', help='variables that are independent from control variables')
    parser.add_argument('--num_time_f', type=int, default=6, help='number of time features')
    parser.add_argument('--is_policy', type=bool, default=True, help='wether is the policy training or not')

    # Parser
    args = parser.parse_args()
    
    ### DLinear
    args.individual = True # Linear layer for each channel
    
    ### Formers
    args.embed_type = 1 # 0: Default, 1: ALL embeddings, 2: Value + Temporal, 3: Value + Positional, 4: Value
    args.n_heads = 8
    args.d_ff = 2048 # Dimension of fully connected layer
    args.factor = 3 # Attention Factor
    args.activation = 'gelu' # Activation function
    args.output_attention = True # Use attention in encoder
    args.lradj = 'type1' # type of the learning rate adjustment
    
    # For policy
    args.checkpoints = './policy_checkpoints/'
    args.is_policy = True
    
    ### Defining customized args for ALL models
    args.model = 'LSTM'
    args.lstm_type = ''
    args.data = 'custom'
    args.root_path = './datasets/'
    args.data_path = 'wastewater.csv'
    args.data_tag = 'IOPTQCfP'
    args.target = 'IN_METAL_Q'
    args.ctrl_vars = ['IN_METAL_Q', 'T1_O2', 'METAL_Q']
    args.ind_vars = ['TEMPERATURE', 'IN_Q', 'MAX_CF']
    args.num_time_f = 6
    args.loss = 'mse'
    args.dilate_all = True
    args.alpha_dilate = 0.8
    args.gamma_dilate = 0.01
    args.test_ratio = 0.15
    args.seq_len = 90
    args.label_len = 10
    args.pred_len = 1
    args.batch_size = 128
    args.learning_rate = 1e-6
    args.patience = 5
    args.train_epochs = 1
    args.use_amp = False # use automatic mixed precision training 
    args.scale = True
    args.simulator = False
    args.use_multi_gpu = False
    args.devices = '0,1'
    args.do_early_stop = True
    args.do_lr_opt = False
    args.do_predict = True
    
    ### LSTM
    args.time_scaled = 'Unscaled'
    args.in_features = 7 + args.num_time_f # Time specs
    args.out_features = len(args.ctrl_vars)
    args.weight_decay = 1e-9
    args.layer_dim = 1
    args.hidden_dim = 80
    args.dropout = 0 if args.layer_dim == 1 else 0.15
    args.self_supervised = False
    args.random_episode_length = True
    args.mimimum_episode_length = 10
    args.maximum_episode_length = 120
    
    # Formers
    args.freq = 't'
    args.embed = 'timeF'
    # enc input: Number of channels in individual DLinear
    args.enc_in = 5
    args.dec_in = 5
    args.c_out = 5    
    args.e_layers = 2
    args.d_layers = 2
    args.d_model = 128

    
    num_input = (args.in_features if args.model == 'LSTM' else args.enc_in)
    num_output = (args.out_features if args.model == 'LSTM' else args.dec_in)
    # label = (0 if args.model == 'LSTM' else args.label_len)
    nodes = (args.hidden_dim if args.model == 'LSTM' else args.d_model)
    loss_type = '_' + args.loss if args.loss == 'dilate' else ''
    if args.model == 'LSTM':
        layers = f'{args.layer_dim}L'
        lstm_type = '_' + args.lstm_type if args.lstm_type != '' else ''
    else:
        layers = f'{args.e_layers}EncL_{args.d_layers}DecL'
        lstm_type = ''
    
    prediction_length = '' if args.pred_len == 1 else f'_{args.pred_len}Pred'
    
    setting = '{}{}_{}_{}F_{}Out_{}_{}_{}Seq{}_{}Batch_{}LR_{}H_'.format(
        args.model, lstm_type, args.data_tag, num_input, num_output, args.embed, args.time_scaled, 
        args.seq_len, prediction_length, args.batch_size, args.learning_rate, nodes) + layers + loss_type
            
    args.setting = setting
    
    # GPU configuration
    if args.use_gpu:
        torch.cuda.empty_cache()

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    ARGS_PATH = './policy_args/' + args.setting + '/'
    if not os.path.exists(ARGS_PATH):    
        os.makedirs(ARGS_PATH)
        
    with open(ARGS_PATH + 'args.pkl', 'wb') as file:
        pickle.dump(args, file)
    
    return args

#%% Training the models

if __name__ ==  '__main__':
   
    args = make_args()
    
    # Setting up the device
    def acquire_device():
        if args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                args.gpu) if not args.use_multi_gpu else args.devices
            device = torch.device('cuda:{}'.format(args.gpu))
            print('Use GPU: cuda:{}'.format(args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    device = acquire_device()
    
    df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
    df_raw = df_raw.set_index(["date"])
    df_raw.index = pd.to_datetime(df_raw.index)
    start_date = pd.to_datetime(df_raw.index[-args.seq_len])
    df_predict = df_raw.loc[start_date:]
    df_raw.rename_axis('date').reset_index(drop = True, inplace = True)
       
    print(args.setting)
    
    if args.model == 'LSTM':
        Exp = ExpLSTM      
        exp = Exp(args, device, is_policy=True)
    else:    
        Exp = Exp_Main
        exp = Exp(df_predict, args)
        
    if args.is_training:
        for ii in range(args.itr):
            # Training the model
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.setting))
            if args.model == 'LSTM' and args.self_supervised:
                exp.self_supervised_train(args.setting)
            else:
                exp.train(args.setting)
                # dilate_ys = exp.opt.dilate_ys
            
            # Testing the model
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.setting))
            exp.test(args.setting)

            if args.do_predict:
                # Predecting the future
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.setting))
                if args.model == 'LSTM':
                    exp.predict_simulation()
                    pred_dict = exp.pred_results_dict
                else:
                    exp.predict(args.setting, True)
                    
                folder_path = './results/' + args.setting + '/'
                # predicted = np.load(folder_path + 'real_prediction.npy')
                
            torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.setting))
        exp.test(args.setting, test=1)
        torch.cuda.empty_cache()     
