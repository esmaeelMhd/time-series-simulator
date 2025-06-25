"""
Created on Monday July 01 2023
@author: Esmaeel Mohammadi

# =============================================================================
# This class is used to handle the retrainer for a saved model improvements:
    1. Training the model
    2. Validating the train results
    3. Prediction using the test dataset
    4. Prediction of the future
    5. Retraining the model using its own prediction
    6. Validation of the retrain results
    7. Test the simulation environment for the retrained model
# =============================================================================
"""

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch
from exp.exp_main import Exp_Main
from utils.env_helper import EnvHelper
from utils.LSTM_model_optimizer import Optimization
from models.LSTM import LSTMModel, EncoderLSTM, DecoderLSTM, Net_LSTM
from models import DLinear
from models import Autoformer
from models import Transformer
from models import Informer
from models import NLinear
import numpy as np
import pandas as pd
import datetime
import os
import re
import logging
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# %%


class Retrainer():
    def __init__(self, args, df, retrain_args, device):
        self.args = args
        self.df = df
        self.retrain_args = retrain_args
        
        # Setup the device based on GPU usage
        self.device = torch.device('cuda') if torch.cuda.is_available() and self.args.use_gpu else torch.device('cpu')
        if self.device == 'cuda':
            self.device_ids = ','.join(str(i) for i in range(torch.cuda.device_count()))
            self.use_multi_gpu = torch.cuda.device_count() > 1
        else:
            self.device_ids = ''
            self.use_multi_gpu = False 
            
        home_dir = os.path.expanduser("~")
        if not os.path.exists(home_dir + '/raid/vz75cp/'):
            dataset_dir = './datasets/'
        else:
            print('Loading from raid ...')
            dataset_dir = os.path.join(home_dir, 'raid/vz75cp/datasets/')

        self.df_raw = pd.read_csv(dataset_dir + self.args.data_path)
        self.df_raw = self.df_raw.set_index(["date"])
        self.df_raw.index = pd.to_datetime(self.df_raw.index)
        if not self.df_raw.index.is_monotonic_increasing:
            self.df_raw = self.df_raw.sort_index()
        '''
        self.results_dictionary = {'data':[],
                                   'targets':[],
                                   'actual_states':[],
                                   'df_raw_scaled':[],
                                   'val_data':[],
                                   'val_targets':[]}
        '''
        self.train_dictionary = []

        # Start by setting up the retrain experiment
        self.set_experiment()

    # Sets the necessary parameters and the name of the experiment
    def set_experiment(self):
        experiment = self.retrain_args.experiment
        # Experiment 1
        if experiment == 1:
            self.random_episode_start = False
            self.random_episode_length = False
            self.const_episode_length = self.retrain_args.const_episode_length

        # Experiment 2
        elif experiment == 2:
            self.random_episode_start = False
            self.random_episode_length = True
            self.min_episode_length = self.retrain_args.min_episode_length
            self.max_episode_length = self.retrain_args.max_episode_length

        # Experiment 3
        elif experiment == 3:
            self.random_episode_start = True
            self.random_episode_length = False
            self.const_episode_length = self.retrain_args.const_episode_length

        # Experiment 4
        elif experiment == 4:
            self.random_episode_start = True
            self.random_episode_length = True
            self.min_episode_length = self.retrain_args.min_episode_length
            self.max_episode_length = self.retrain_args.max_episode_length

        gap_flag = self.retrain_args.gap_flag
        assert gap_flag in ['no_gap', 'small_gap', 'large_gap', 'custom_gap']
        gap_map = {'no_gap': 'No_Gap', 'small_gap': 'SGap', 'large_gap': 'LGap',
                   'custom_gap': f'{self.retrain_args.custom_gap}Gap'}
        self.gap_name = gap_map[gap_flag]
        self.gap = gap_flag

        files = list([name for name in os.listdir(
            './checkpoints/' + self.retrain_args.setting + '/')])
        prefix = f'E{experiment}N'
        ex_files = [f for f in files if f.startswith(prefix)]
        max_num = 0
        for f in ex_files:
            match = re.search(f'{prefix}(\d+)', f)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
        self.retrain_number = max_num + 1

        actual_actions = 'ActA' if self.retrain_args.use_actual_actions else 'No_ActA'
        ss_setting = '_SS' if self.retrain_args.scheduled_sampling else ''
        dilate_all = '_all' if (
            self.retrain_args.dilate_all and self.retrain_args.loss_function == 'dilate') else ''
        loss_type = f'{self.retrain_args.loss_function}{dilate_all}_{self.retrain_args.alpha_dilate}A_' if self.retrain_args.loss_function == 'dilate' else ''
        val_loss_type = '_V' + \
            self.retrain_args.val_loss_type if self.retrain_args.val_loss_type != 'mse' else ''

        if self.retrain_args.retrain_method == 'sepp':
            retr_method = 'SEPP'
            max_el = f'{self.max_episode_length}MaxEL' if self.random_episode_length else f'{self.const_episode_length}EL'
            retr_mode = 'Batch' if self.retrain_args.retrain_mode == 'batch_pred' else 'Single'
            small_batch_str = f'_{self.retrain_args.num_small_batches}SB' if (
                experiment == 3 or experiment == 4) else ''
            sepp_gap = self.gap_name

            self.retrain_name = '{}_E{}N{} - {}Ep_{}_{}_{}_{}{}_{}{}LR{}{}'.format(retr_method, experiment, self.retrain_number,
                                                                                self.retrain_args.retrain_epochs, actual_actions,
                                                                                sepp_gap, max_el, retr_mode, small_batch_str,
                                                                                loss_type, self.retrain_args.retrain_lr, ss_setting, val_loss_type)
        else:
            retr_method = self.retrain_args.retrain_method
            max_el = f'{self.max_episode_length}MaxEL' if self.random_episode_length else f'{self.const_episode_length}EL'
            retr_mode = 'Batch' if self.retrain_args.retrain_mode == 'batch_pred' else 'Single'
            small_batch_str = f'_{self.retrain_args.num_small_batches}SB' if (
                experiment == 3 or experiment == 4) else ''

            self.retrain_name = 'E{}N{} - {}Ep_{}_{}_{}_{}{}_{}{}LR{}{}'.format(experiment, self.retrain_number,
                                                                                self.retrain_args.retrain_epochs, actual_actions,
                                                                                self.gap_name, max_el, retr_mode, small_batch_str,
                                                                                loss_type, self.retrain_args.retrain_lr, ss_setting, val_loss_type)
    # Builds the model

    def build_model(self):
        model_dict = {
            'LSTM': LSTMModel,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear
        }

        if self.args.model == 'LSTM':
            if not hasattr(self.args, 'lstm_type') or (hasattr(self.args, 'lstm_type') and self.args.lstm_type != 'EncDec'):
                model = LSTMModel(self.args).float()
            else:
                encoder = EncoderLSTM(self.args)
                decoder = DecoderLSTM(self.args)
                model = Net_LSTM(encoder, decoder, self.args,
                                 self.device).float()
        else:
            model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # Splits the dataset according to the type of the experiment
    def data_split(self, df, experiment, val=False):
        X = []
        y = []
        start_idx = 0

        if val == False:
            # Sets the initial index and episode length
            if experiment == 1 or experiment == 3:
                episode_length = self.retrain_args.const_episode_length
            elif experiment == 2 or experiment == 4:
                episode_length = np.random.randint(self.retrain_args.min_episode_length,
                                                   self.retrain_args.max_episode_length)

            # Iterates through the dataset and builds the X, and y lists
            while start_idx + self.retrain_args.seq_length + episode_length < len(df):
                end_idx = start_idx + self.retrain_args.seq_length
                X.append(df[start_idx: end_idx])
                y.append(
                    df[end_idx: end_idx + episode_length + self.args.pred_len-1])

                if self.gap != 'no_gap':
                    if self.gap == 'small_gap':
                        start_idx = start_idx + episode_length
                    elif self.gap == 'large_gap':
                        start_idx = start_idx + self.retrain_args.seq_length + episode_length
                    else:
                        start_idx = start_idx + self.retrain_args.custom_gap
                else:
                    start_idx = start_idx + 1

                if experiment == 2 or experiment == 4:
                    episode_length = np.random.randint(self.retrain_args.min_episode_length,
                                                       self.retrain_args.max_episode_length)

        # Builds the validation dataset
        else:
            episode_length = self.retrain_args.val_length
            while start_idx + self.retrain_args.seq_length + episode_length < len(df):
                end_idx = start_idx + self.retrain_args.seq_length
                X.append(df[start_idx: end_idx])
                y.append(
                    df[end_idx: end_idx + episode_length + self.args.pred_len-1])
                start_idx = start_idx + self.retrain_args.seq_length + episode_length

        return (X, y)

    # A method to do random indexing of the dataset
    def shuffle_with_fixed_chunks(self, data, targets, chunk_size):
        # Divides the list into chunks
        idxs = list(range(0, len(data)-chunk_size, chunk_size))
        np.random.shuffle(idxs)

        shuffled_data = []
        shuffled_targets = []
        chunks = []
        for idx in idxs:
            for i in range(chunk_size):
                shuffled_data.append(data[idx+i])
                shuffled_targets.append(targets[idx+i])
                chunks.append(idx+i)

        return shuffled_data, shuffled_targets, chunks

    # Loads the saved model checkpoint and builds the optimizer if needed
    def load_model(self, checkpoint_path):
        # Loading the model
        print('loading the model ...')
        self.loaded_model = self.build_model().to(self.device)
        if os.path.isfile(os.path.join(checkpoint_path, self.retrain_name + '.pth')):
            print('Continuing from the previous checkpoint:')
            self.loaded_model.load_state_dict(torch.load(
                checkpoint_path + self.retrain_name + '.pth'))
        else:
            self.loaded_model.load_state_dict(
                torch.load(os.path.join(checkpoint_path ,'checkpoint.pth'), map_location=self.device))

        if self.args.model == 'LSTM':
            # Building the optimizer
            loss_fn = nn.MSELoss(reduction="mean")

            optimizer = optim.Adam(self.loaded_model.parameters(), lr=self.retrain_args.retrain_lr,
                                   weight_decay=self.retrain_args.retrain_wd)

            self.opt = Optimization(self, model=self.loaded_model, loss_fn=loss_fn, optimizer=optimizer,
                                    args=self.args, setting=self.args.setting, device=self.device, is_retrain=True)
        else:
            self.exp = Exp_Main(self.df, self.args)

        print(
            f'Retraining will be done for {self.args.model}: ' + self.retrain_name)

    # Prepares the inputs and targets for retraining
    def prepare_data(self):
        experiment = self.retrain_args.experiment
        # Gets the simulation test points and names
        self.helper = EnvHelper()
        test_points, self.test_points_names = self.helper.make_points(test_frequency='Seasons', time_of_the_day='Morning',
                                                                      day_of_the_month='Middle', first_date=pd.to_datetime('2021-08'),
                                                                      last_date=pd.to_datetime('2022-07'))
        # Add time specs to the whole raw dataset
        self.df = self.helper.add_time_specs(self.df)
        # Scales the whole dataset
        if self.args.scale:
            self.df = self.helper.scale_data(self.args, self.df)

        df_train, df_val = train_test_split(self.df.astype(
            np.float32), test_size=self.retrain_args.val_ratio, shuffle=False)

        # Splits the whole datasets to different episodes (batches) to retrain
        self.train_data, self.train_targets = self.data_split(
            df_train, experiment)
        val_data, val_targets = self.data_split(df_val, experiment, val=True)
        # self.results_dictionary['val_data'].append(np.array(val_data))
        # self.results_dictionary['val_targets'].append(np.array(val_targets))

        # Creates the test loader
        test_data = []
        test_targets = []
        for point in test_points:
            test_start = self.df_raw.index.get_loc(point)
            test_end = test_start + self.args.seq_len
            X = self.helper.scale_data(self.args, self.helper.add_time_specs(
                self.df_raw[test_start:test_end]))
            y = self.helper.scale_data(self.args, self.helper.add_time_specs(
                self.df_raw[test_end:test_end + self.retrain_args.val_length]))
            test_data.append(X)
            test_targets.append(y)

        test_data = torch.Tensor(np.array(test_data)).to(self.device)
        test_targets = torch.Tensor(np.array(test_targets)).to(self.device)
        test_dataset = TensorDataset(test_data, test_targets)
        self.test_loader = DataLoader(test_dataset, batch_size=1,
                                      shuffle=False, drop_last=True)

        # Creates the val loader
        val_data = torch.Tensor(np.array(val_data)).to(self.device)
        val_targets = torch.Tensor(np.array(val_targets)).to(self.device)
        val_dataset = TensorDataset(val_data, val_targets)
        self.val_loader = DataLoader(val_dataset, batch_size=1,
                                     shuffle=False, drop_last=True)

        # Creates the random batches according to the experiments
        if experiment == 3 or experiment == 4:
            self.train_data, self.train_targets, self.data_idxs = self.shuffle_with_fixed_chunks(self.train_data,
                                                                                                 self.train_targets,
                                                                                                 self.retrain_args.num_small_batches)

    # Writes the retrain metrics to log and csv files
    def write_metrics(self):
        # Set up logging
        logging.basicConfig(filename='retrain_metrics.log', level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Log the metrics for the current experiment
        metrics_dict = {'Name': self.retrain_name,
                        'Date': str(datetime.datetime.now())}
        metrics_dict['Experiment'] = self.retrain_args.experiment
        metrics_dict['Number'] = self.retrain_number
        metrics_dict['Epochs'] = self.retrain_args.retrain_epochs
        metrics_dict['Actual Actions'] = self.retrain_args.use_actual_actions
        metrics_dict['Use Gap'] = self.gap_name
        metrics_dict['Episode Length'] = self.retrain_args.const_episode_length if (self.retrain_args.experiment == 1
                                                                                    or self.retrain_args.experiment == 3) else self.retrain_args.max_episode_length
        metrics_dict['Retrain Mode'] = self.retrain_args.retrain_mode
        metrics_dict['Small Batches'] = self.retrain_args.num_small_batches if (self.retrain_args.experiment == 3
                                                                                or self.retrain_args.experiment == 4) else 1
        metrics_dict['Use SS'] = True if self.retrain_args.scheduled_sampling else False
        metrics_dict['SS Prob'] = self.retrain_args.ss_prob if self.retrain_args.scheduled_sampling else '-'
        metrics_dict['Loss'] = self.retrain_args.loss_function
        metrics_dict['Best loss'] = self.best_loss
        metrics_dict['Validation loss'] = np.mean(
            self.val_losses) if self.retrain_args.retrain_method == 'v1' else self.best_loss
        metrics_dict['Train loss'] = np.mean(self.train_losses)
        logging.info(metrics_dict)

        # Write metrics to the CSV file
        fieldnames = metrics_dict.keys()

        with open('retrain_metrics.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header if the file is empty
            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow(metrics_dict)

    # Initiates the retraining process
    def start_retrain(self):
        torch.cuda.empty_cache()
        # Loads the previously trained model
        setting = self.args.setting
        checkpoint_path = './checkpoints/' + setting + '/'
        self.load_model(checkpoint_path)
        experiment = self.retrain_args.experiment
        self.data_idxs = None

        if self.args.model == 'LSTM':
            self.best_loss = None
            self.train_losses = []
            self.val_losses = []
            self.prepare_data()
            # self.results_dictionary['train_data'].append(train_data)
            # self.results_dictionary['train_targets'].append(train_targets)

            # Creates a SummaryWriter for TensorBoard logging
            log_dir = checkpoint_path + 'retrain logs' + '/' + self.retrain_name
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # writer = SummaryWriter(log_dir=log_dir)

            # Starts the retrain process
            self.opt.retrain(retrain_args=self.retrain_args, model=self.loaded_model,
                             retrain_name=self.retrain_name, retrain_data=self.train_data,
                             retrain_targets=self.train_targets, data_idxs=self.data_idxs,
                             val_loader=self.val_loader, test_loader=self.test_loader)  # writer=writer

            self.train_losses = self.opt.train_losses
            self.val_losses = self.opt.val_losses
            # dict_t = self.opt.retrain_dict
            # self.train_dictionary.append(dict_t)
            self.best_loss = np.min(self.val_losses)
            # writer.close()

            # plot opt.train_losses
            # plot opt.val_losses

            self.write_metrics()

    def start_retrain_sepp(self):
        torch.cuda.empty_cache()
        # Loads the previously trained model
        setting = self.args.setting
        checkpoint_path = './checkpoints/' + setting + '/'
        self.load_model(checkpoint_path)
        experiment = self.retrain_args.experiment
        self.data_idxs = None

        if self.args.model == 'LSTM':
            self.best_loss = None
            self.train_losses = []
            self.val_losses = []
            self.prepare_data()
            # self.results_dictionary['train_data'].append(train_data)
            # self.results_dictionary['train_targets'].append(train_targets)

            # Creates a SummaryWriter for TensorBoard logging
            log_dir = checkpoint_path + 'retrain logs' + '/' + self.retrain_name
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # writer = SummaryWriter(log_dir=log_dir)

            # Starts the retrain process
            self.opt.retrain_sepp(retrain_args=self.retrain_args, model=self.loaded_model,
                             retrain_name=self.retrain_name, retrain_data=self.train_data,
                             retrain_targets=self.train_targets, data_idxs=self.data_idxs,
                             val_loader=self.val_loader, test_loader=self.test_loader)  # writer=writer

            self.train_losses = self.opt.train_losses
            self.val_losses = self.opt.val_losses
            # dict_t = self.opt.retrain_dict
            # self.train_dictionary.append(dict_t)
            self.best_loss = np.min(self.val_losses)
            # writer.close()

            # plot opt.train_losses
            # plot opt.val_losses

            self.write_metrics()

    # Runs the test simulation for retrained model
    def test_retrain(self):
        self.test_results_dict = {key: None for key in self.test_points_names}
        # Loading the model
        print('Loading model for test ...')
        self.loaded_model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.args.setting,
                                                                  self.retrain_name + '.pth')))

        test_results_list, test_losses = self.opt.test_simulation(
            self.loaded_model, self.test_loader)
        for i, (key, item) in enumerate(zip(self.test_results_dict, test_results_list)):
            self.test_results_dict[key] = item
            y_real = self.test_results_dict[key]['y_real']
            y_pred = self.test_results_dict[key]['y_pred']
            self.test_results_dict[key]['y_real'] = self.helper.inverse_transform(
                y_real)
            self.test_results_dict[key]['y_pred'] = self.helper.inverse_transform(
                y_pred)

        test_loss = np.mean(test_losses)

        fig_dpi = 1000
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), dpi=fig_dpi)
        fig.tight_layout(pad=4.0)

        if self.args.target == 'T1_PO4':
            ylabel = 'P-concentration (mg/L)'
        elif self.args.target == 'N2O':
            ylabel = 'N2O'

        for i, (ax, point) in enumerate(zip(axs.ravel(), self.test_results_dict.keys())):
            test_real = self.test_results_dict[point].get('y_real')
            test_pred = self.test_results_dict[point].get('y_pred')
            ax.plot(test_real[:, -1], label='Ground Truth', color='black')
            ax.plot(test_pred[:, -1], label='Prediction')
            ax.set_title(point, fontweight="bold")
            ax.set_ylabel(ylabel, labelpad=10)
            ax.set_xlabel('Step', labelpad=10)

        '''
        axs[1].plot(retrainer.train_losses, label='Training Loss')
        axs[1].plot(retrainer.val_losses, label='Validation Loss')
        axs[1].set_title('Training Results', fontweight="bold")
        axs[1].set_ylabel('Loss', labelpad=10)
        axs[1].set_xlabel('Batch', labelpad=10)
        '''

        plt.legend()

        RETRAIN_RESULTS_PATH = './retrain_results/'
        fig_folder_path = RETRAIN_RESULTS_PATH + self.retrain_args.setting + '/'
        if not os.path.exists(fig_folder_path):
            os.makedirs(fig_folder_path)

        fig_name = 'Test - ' + self.retrain_name

        plt.savefig(fig_folder_path + fig_name + '.svg')
        plt.savefig(fig_folder_path + fig_name + '.png')
        plt.savefig(fig_folder_path + fig_name + '.pdf')

        return
