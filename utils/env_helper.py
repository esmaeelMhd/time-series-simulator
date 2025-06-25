from tslearn.metrics import dtw, dtw_path
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from numpy import zeros, newaxis
import copy
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
from dateutil.relativedelta import relativedelta
import pytz
import joblib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

class EnvHelper():
    def __init__(self):
        self = self

        self.feature_scaler = None
        self.time_scaler = None
        self.scaler = None
        self.freq = datetime.timedelta(minutes=1)
        self.legend_size = 10

        try:
            plt.rcParams['font.family'] = 'Times New Roman'
        except Exception as e:
            pass
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.linewidth'] = 0.5
        plt.rcParams['axes.xmargin'] = 0.02
        plt.rcParams['axes.ymargin'] = 0.04
        plt.rcParams['axes.labelsize'] = 8

        plt.rc('axes', titlesize=8)
        plt.rc('axes', labelsize=8)
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('legend', fontsize=self.legend_size)
        
        self.fig_dpi = 1000
        self.fig_width = 6.2

    def make_points(self, test_frequency='Months', day_of_the_month='Middle',
                    time_of_the_day='Morning', first_date=pd.to_datetime('2021-08'),
                    last_date=pd.to_datetime('2022-12')):

        self.time_of_the_day = time_of_the_day

        def season_of_date(date):
            year = date.year
            seasons = {'Summer': (datetime.datetime(year, 6, 21), datetime.datetime(year, 9, 22)),
                       'Autumn': (datetime.datetime(year, 9, 23), datetime.datetime(year, 12, 20)),
                       'Spring': (datetime.datetime(year, 3, 21), datetime.datetime(year, 6, 20))}
            for season, (season_start, season_end) in seasons.items():
                season_start = season_start.replace(tzinfo=pytz.UTC)
                season_end = season_end.replace(tzinfo=pytz.UTC)
                if date >= season_start and date <= season_end:
                    return season
            else:
                return 'Winter'

        def create_points():
            freq_values = {'Months': 1, 'Seasons': 3}
            day_values = {'First': 1, 'Middle': 15}
            time_values = {'Morning': 0, 'Noon': 12}

            start_date = datetime.datetime(first_date.year, first_date.month, day_values[day_of_the_month],
                                           time_values[time_of_the_day], 0, 0).replace(tzinfo=pytz.UTC)
            end_date = datetime.datetime(last_date.year, last_date.month, day_values[day_of_the_month],
                                         time_values[time_of_the_day], 0, 0).replace(tzinfo=pytz.UTC)

            month_freq = relativedelta(months=freq_values[test_frequency])
            list_points = []
            points_names = []
            date = start_date

            while date <= end_date:
                date = date.replace(tzinfo=pytz.UTC)
                list_points.append(date)
                if test_frequency == 'Seasons':
                    # season = season_of_date(date)
                    points_names.append(f'{date.strftime("%b %d %Y")}')
                else:
                    points_names.append(f'{date.strftime("%b %d %Y")}')

                date += month_freq
            return list_points, points_names

        # Creating the points dataframe and names
        self.list_points, self.points_names = create_points()

        return self.list_points, self.points_names

    def make_models_list(self, setting='', single=False, best_test_results=True,
                         best_env_results=False, model_based=False, dataset_based=False,
                         all_models=False, model_name='LSTM', data_tag='NewCorrH', episode_length=180):

        # Information about the models
        result_folders = list([name for name in os.listdir('./results/')])
        list_metrics = list(['mae', 'mse', 'rmse', 'corr', 'r2'])
        df_metrics = pd.DataFrame(columns=list_metrics)
        df_metrics['names'] = result_folders
        df_metrics = df_metrics.set_index(['names'])

        # Handling the metrics dataframe
        for folder in result_folders:
            if 'metrics.npy' in os.listdir('./results/' + folder):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    metrics = np.load('./results/' + folder +
                                      '/metrics.npy', allow_pickle=True)
                    # pred = np.load('./results/' + folder + '/pred.npy', allow_pickle=True)
                    # true = np.load('./results/' + folder + '/true.npy', allow_pickle=True)
                    # pred = pred[:,:,-1].reshape(pred.shape[0], 1)
                    # true = true[:,:,-1].reshape(pred.shape[0], 1)
                if len(metrics) == 7:
                    # Deleting the unnecessary metrics
                    metrics = np.delete(metrics, 3)
                    metrics = np.delete(metrics, 4)
                # metrics = metrics.astype(float)
                df_metrics.loc[folder, ['mae', 'mse', 'rmse']] = metrics[:3]
                df_metrics.loc[folder, 'corr'] = np.mean(metrics[-1])
                # df_metrics.loc[folder, 'r2'] = r2_score(true, pred)

        best_models = {'LSTM': '',
                       'Transformer': '',
                       'Informer': '',
                       'Autoformer': '',
                       'DLinear': '',
                       'NLinear': '',
                       }

        models_list = []

        if single:
            models_list.append(setting)

        elif best_test_results:
            if dataset_based:
                folders = list([x for x in result_folders if any(
                    item in x for item in data_tag)])
            else:
                folders = result_folders

            for model in best_models.keys():
                items = list([x for x in folders if model in x])
                df_metrics['mse'] = pd.to_numeric(df_metrics['mse'])
                list_mse = df_metrics.loc[items, 'mse']
                min_loss_idx = list_mse.idxmin()
                best_models[model] = min_loss_idx
            models_list = best_models.values()

        elif best_env_results:
            if dataset_based:
                folders = list([x for x in os.listdir(
                    './env_results/') if any(item in x for item in data_tag)])
            else:
                folders = list([x for x in os.listdir('./env_results/')])

            p_names = self.points_names
            df_mse = pd.DataFrame(columns=p_names)
            df_mse['Average'] = None
            df_mse['names'] = folders
            df_mse = df_mse.set_index(['names'])

            for folder in folders:
                for point_name in p_names:
                    RESULTS_PATH = './env_results/' + folder + '/'
                    try:
                        # print(f'file name: {episode_length}_rounds_{point_name} - {self.time_of_the_day}_states.npy')
                        states = np.load(
                            RESULTS_PATH + f'{episode_length}_rounds_{point_name} - {self.time_of_the_day}_Phosphorous.npy')
                        actual_states = np.load(
                            RESULTS_PATH + f'{episode_length}_rounds_{point_name} - {self.time_of_the_day}_actual_p.npy')

                        states = np.array(states).reshape(
                            np.array(states).shape[0], -1)
                        actual_states = np.array(actual_states).reshape(
                            np.array(actual_states).shape[0], -1)
                        states[:, 0] = actual_states[:, 0]

                        mse = mean_squared_error(
                            actual_states[:, -1], states[:, -1])
                        df_mse.loc[folder, point_name] = mse

                    except FileNotFoundError:
                        print('files not found!')

            df_mse['Average'] = df_mse.mean(axis=1)
            df_mse = df_mse.sort_values('Average')

            for model in best_models.keys():
                items = list([x for x in df_mse.index if model in x])
                list_mse = df_mse.loc[items, 'Average']
                max_mse_idx = df_mse.loc[items, 'Average'].idxmin()
                best_models[model] = max_mse_idx

            models_list = best_models.values()

        elif model_based:
            folders = list([x for x in result_folders if model_name in x])
            if dataset_based:
                models_list = list(
                    [x for x in folders if any(item in x for item in data_tag)])
            else:
                models_list = folders

        elif all_models:
            if dataset_based:
                models_list = list(
                    [x for x in result_folders if any(item in x for item in data_tag)])
            else:
                models_list = result_folders

        df_metrics = df_metrics[df_metrics.index.isin(models_list)]
        if best_env_results or best_test_results:
            for key in best_models.keys():
                for model in df_metrics.index:
                    if best_models[key] == model:
                        df_metrics = df_metrics.rename(index={model: key})

        # df_metrics = df_metrics.drop(columns=['names', 'models'])
        custom_order = [key for key in best_models.keys()]
        df_metrics.index = pd.Categorical(
            df_metrics.index, categories=custom_order, ordered=True)
        df_metrics = df_metrics.sort_index()
        df_metrics = df_metrics.astype(float)
        df_metrics = df_metrics.round(4)

        return models_list, best_models, df_metrics

    def make_retrained_list(self, setting, chkpt_name='', single_retr=False,
                            experiment_based=False, experiments=['E1'], number_based=False,
                            number_list=[], all_retrained=True):
        checkpoints_path = './checkpoints/' + setting + '/'
        checkpoints_list = list(
            [name for name in os.listdir(checkpoints_path)])
        retrained_list = []
        experiments_list = ['E1', 'E2', 'E3', 'E4']
        if single_retr:
            retrained_list.append(chkpt_name)
        elif experiment_based:
            experiments_list = experiments
            retrained_list = list([x for x in checkpoints_list if any(
                item in x for item in experiments_list)])
        elif number_based:
            retrained_list = list(
                [x for x in checkpoints_list if any(item in x for item in number_list)])
        elif all_retrained:
            retrained_list = list([x for x in checkpoints_list if any(
                item in x for item in experiments_list)])

        retrained_names = []
        for checkpoint in retrained_list:
            words = checkpoint.split()
            if 'final' in checkpoint:
                chkpt_name = words[0] + ' - Final'
            else:
                chkpt_name = words[0] + ' - Best loss'

            if 'mse' in checkpoint:
                chkpt_name += '_mse'
            elif 'mae' in checkpoint:
                chkpt_name += '_mae'

            retrained_names.append(chkpt_name)

        return retrained_list, retrained_names

    def scale_data(self, args, df, normalize_env_results=False):
        self.args = args
        self.num_cols = self.args.out_features if args.model == 'LSTM' else self.args.enc_in

        # Load the scaler path
        if hasattr(self.args, 'is_policy') and self.args.is_policy:
            SCALER_PATH = './policy_scalers/' + self.args.setting + '/'
        else:
            SCALER_PATH = './scalers/' + self.args.setting + '/'

        if self.args.model == 'LSTM':
            if self.args.time_scaled == 'Unscaled':
                if self.feature_scaler is None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        self.feature_scaler = joblib.load(
                            SCALER_PATH + 'feature_scaler.gz')
            else:
                if self.feature_scaler is None or self.time_scaler is None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        self.feature_scaler = joblib.load(
                            SCALER_PATH + 'feature_scaler.gz')
                        self.time_scaler = joblib.load(
                            SCALER_PATH + 'time_scaler.gz')
        else:
            if self.scaler is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    self.scaler = joblib.load(SCALER_PATH + 'scaler.gz')

        num_features = len(self.feature_scaler.scale_)

        if self.args.model == 'LSTM':
            try:
                if normalize_env_results:
                    arr = self.feature_scaler.transform(df)
                else:
                    arr = np.zeros(shape=(df.shape[0], self.args.in_features))
                    if len(df.columns) == num_features:
                        arr = self.feature_scaler.transform(df)
                    else:
                        if self.args.time_scaled == 'Scaled':
                            # Scale the feature columns
                            arr[:, :num_features] = self.feature_scaler.transform(
                                df.iloc[:, :num_features])
                            # Scale the time columns
                            arr[:, num_features:] = self.time_scaler.transform(
                                df.iloc[:, num_features:])
                        else:
                            # Scale only the feature columns
                            arr[:, :num_features] = self.feature_scaler.transform(
                                df.iloc[:, :num_features])
                            arr[:, num_features:] = np.array(
                                df.iloc[:, num_features:])
            except ValueError as e:
                print("ValueError occurred:", e)
                print(
                    f'Scaler num features: {num_features} and In variables: {self.args.in_features - 6}')
        else:
            arr = self.scaler.transform(df)

        return arr

    # Function for the inverse transform of the scaled data
    def inverse_transform(self, arr):
        num_features = len(self.feature_scaler.scale_)
        if arr.shape[-1] < num_features:
            arr = np.hstack(
                (np.zeros((len(arr), num_features - arr.shape[-1])), arr))
        if self.args.model == 'LSTM':
            if (
                self.args.embed == 'timeF'
                and self.args.time_scaled == 'Unscaled'
                or self.args.embed != 'timeF'
            ):
                arr = self.feature_scaler.inverse_transform(arr)
            else:
                arr = self.feature_scaler.inverse_transform(arr)

            if arr.shape[-1] < num_features:
                arr = arr[:, num_features - arr.shape[-1]:]

        else:
            arr = self.scaler.inverse_transform(arr)
        return arr

    # Converting the arrays to dataframe, we need to do it because of the scaler
    # and also making Tensor dataset
    def make_df(self, arr, step):
        start = self.start_date
        first_date = start + (step)*self.freq
        index = pd.date_range(
            start=first_date, freq=self.freq, periods=len(arr))
        df = pd.DataFrame(arr, columns=self.df.columns[:self.num_cols])
        df = df.set_index(index)
        if self.args.model == 'LSTM' and self.args.embed == 'timeF':
            df = self._add_time_specs(df)
        return df

    # Addition of Time Specifications if we need them
    def add_time_specs(self, df):
        # Cyclical features generator
        def generate_cyclical_features(df, col_name, period, start_num=0):
            kwargs = {
                f'sin_{col_name}': lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
                f'cos_{col_name}': lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)
            }

            return df.assign(**kwargs).drop(columns=[col_name])

        # Add time specs
        df = (df
              .assign(hour=df.index.hour)
              .assign(day_of_week=df.index.dayofweek)
              .assign(month=df.index.month)
              )

        # Convert time specs to sin and cos
        df = generate_cyclical_features(df, 'hour', 24, 0)
        df = generate_cyclical_features(df, 'day_of_week', 7, 0)
        df = generate_cyclical_features(df, 'month', 12, 1)

        return df

    def best_retrain_results(self, model_path, retrained_list=[], el=180):

        best_retrained = {'E1': '', 'E2': '', 'E3': '', 'E4': ''}
        retrained_list = [string.replace('.pth', '')
                          for string in retrained_list]

        p_names = self.points_names
        df_mse = pd.DataFrame(columns=p_names)
        df_mse['Average'] = None
        df_mse['names'] = retrained_list
        df_mse = df_mse.set_index(['names'])

        for retrained in retrained_list:
            for point_name in p_names:
                RESULTS_PATH = './env_results/' + model_path + '/'

                if retrained == 'checkpoint.pth' or retrained == 'checkpoint':
                    file_name = ''
                else:
                    file_name = retrained + '_'
                try:
                    # print(f'file name: {episode_length}_rounds_{point_name} - {self.time_of_the_day}_states.npy')
                    states = np.load(
                        RESULTS_PATH + file_name + f'{el}_rounds_{point_name}_{self.time_of_the_day}_states.npy')
                    actual_states = np.load(
                        RESULTS_PATH + file_name + f'{el}_rounds_{point_name}_{self.time_of_the_day}_actual_states.npy')

                    states = np.array(states).reshape(
                        np.array(states).shape[0], -1)
                    actual_states = np.array(actual_states).reshape(
                        np.array(actual_states).shape[0], -1)
                    states[:, 0] = actual_states[:, 0]

                    mse = mean_squared_error(
                        actual_states[:, -1], states[:, -1])
                    df_mse.loc[retrained, point_name] = mse

                except FileNotFoundError:
                    print('files not found!')

        df_mse['Average'] = df_mse.mean(axis=1)
        df_mse = df_mse.sort_values('Average')

        for experiment in best_retrained.keys():
            items = list([x for x in df_mse.index if experiment in x])
            list_mse = df_mse.loc[items, 'Average']
            max_mse_idx = df_mse.loc[items, 'Average'].idxmin()
            best_retrained[experiment] = max_mse_idx

        return best_retrained

    def compute_dtw(self, real, pred):
        if isinstance(real, torch.Tensor):
            if real.ndim == 2:
                real = real.unsquueze(0)
            elif real.ndim == 1:
                real = real.view(1, len(real), 1)
            if pred.ndim == 2:
                pred = pred.unsquueze(0)
            elif pred.ndim == 1:
                pred = pred.view(1, len(real), 1)
        elif isinstance(real, np.ndarray):
            if real.ndim == 2:
                real = real[newaxis, :, :]
            elif real.ndim == 1:
                real = real[newaxis, :, newaxis]
            if pred.ndim == 2:
                pred = pred[newaxis, :, :]
            elif pred.ndim == 1:
                pred = pred[newaxis, :, newaxis]
            real = torch.Tensor(real)
            pred = torch.Tensor(pred)

        # DTW and TDI
        loss_dtw, loss_tdi = 0.0, 0.0
        for k in range(1):
            batch_size, N_output = pred[:, :, -
                                        1].reshape([1, pred.shape[1], 1]).shape[0:2]
            target = real[:, :, -1].reshape([1, pred.shape[1], 1])
            outputs = pred[:, :, -1].reshape([1, pred.shape[1], 1])
            target = torch.Tensor(target)
            outputs = torch.Tensor(outputs)
            target_k_cpu = target[k, :, 0:1].view(-1).detach().cpu().numpy()
            output_k_cpu = outputs[k, :, 0:1].view(-1).detach().cpu().numpy()

            path, sim = dtw_path(target_k_cpu, output_k_cpu)
            loss_dtw += sim

            Dist = 0
            for i, j in path:
                Dist += (i-j)*(i-j)
            loss_tdi += Dist / (N_output*N_output)

        loss_dtw = loss_dtw / batch_size
        loss_tdi = loss_tdi / batch_size

        return loss_dtw, loss_tdi

    def check_missing_tests(self, args, chkpt_list, start_date, end_date, episode_length):
        missing = []
        for chkpt in chkpt_list:
            dict_name = f'{episode_length}EL_' + chkpt + '_' + f'{start_date.strftime("%b %d %Y")}' +\
                        '_to_' + f'{end_date.strftime("%b %d %Y")}'
            file_path = './env_results/' + args.setting + '/' + dict_name + '.pkl'
            if not os.path.exists(file_path):
                missing.append(chkpt)
                '''
            else:
                with open('./env_results/' + args.setting + '/' + dict_name + '.pkl', 'rb') as file:
                    results = pickle.load(file)
                for point_dict in results.values():
                    if not 'loss_dtw' in point_dict.keys():
                        point_dict = self.add_dtw(point_dict)
                        
                with open('./env_results/' + args.setting + '/' + dict_name + '.pkl', 'wb') as file:
                    pickle.dump(results, file)
        '''
        print(f'There are {len(missing)} missing env test results.')
        return missing



    def add_dtw(self, results_dict):
        y_real = torch.tensor(results_dict['y_real'])
        y_real = y_real.view([1, y_real.shape[0], y_real.shape[1]])
        y_pred = torch.tensor(results_dict['y_pred'])
        y_pred = y_pred.view([1, y_pred.shape[0], y_pred.shape[1]])

        results_dict['loss_dtw'], results_dict['loss_tdi'] = self.compute_dtw(
            y_real, y_pred)

        return results_dict

    def find_best_results(self, args, chkpt_list, chkpt_names, ex_list, points_names,
                          start_date, end_date, loss_type='mse', alpha=0.5, ep_len=1440):

        # Remove the base model
        chkpt_list.pop(0)
        chkpt_names.pop(0)

        ex_list.append('Base Model')
        best_retrained = {key: None for key in ex_list}
        all_loss_dict = {key: None for key in ex_list}
        retrained_list = [string.replace('.pth', '') for string in chkpt_list]

        df_loss = pd.DataFrame(columns=points_names)
        df_loss['names'] = ['Base Model'] + retrained_list
        df_loss = df_loss.set_index(['names'])

        df_mse = copy.deepcopy(df_loss)
        df_dtw = copy.deepcopy(df_loss)

        global df_data
        df_data = pd.DataFrame()
        df_data['Datetime'] = points_names
        df_data = df_data.set_index('Datetime')

        def add_base_model():
            base_model = 'Base'
            dict_name = f'{ep_len}EL_' + base_model + '_' + f'{start_date.strftime("%b %d %Y")}' +\
                        '_to_' + f'{end_date.strftime("%b %d %Y")}'
            try:
                with open('./env_results/' + args.setting + '/' + dict_name + '.pkl', 'rb') as file:
                    base_results = pickle.load(file)

                for key in base_results.keys():
                    p_mse = mean_squared_error(base_results[key]['y_real'][:, -1],
                                               base_results[key]['y_pred'][:, -1])

                    df_data.loc[key, 'Base Model_mse'] = p_mse
                    df_data.loc[key,
                                'Base Model_dtw'] = base_results[key]['loss_dtw']

                    if loss_type == 'mse':
                        df_loss.loc['Base Model', key] = p_mse
                        # df_loss.loc['Base Model', key] = base_results[key]['test_loss']
                    elif loss_type == 'dtw':
                        df_loss.loc['Base Model',
                                    key] = base_results[key]['loss_dtw']

                    if loss_type == 'both':
                        df_mse.loc['Base Model', key] = p_mse
                        # df_mse.loc['Base Model', key] = base_results[key]['test_loss']
                        df_dtw.loc['Base Model',
                                   key] = base_results[key]['loss_dtw']

                best_retrained['Base Model'] = {
                    'name': 'Base Model',
                    'loss': df_loss.loc['Base Model'].mean(),
                    'mse': df_data['Base Model_mse'].mean(),
                    'dtw': df_data['Base Model_dtw'].mean()}

            except FileNotFoundError:
                print(f'File for {dict_name} not found.')
                best_retrained['Base Model'] = {
                    'name': 'Base Model',
                    'loss': None,
                    'mse': None,
                    'dtw': None}

        add_base_model()

        def add_experiments():
            for i, checkpoint in enumerate(retrained_list):
                chkpt_name = chkpt_names[i]
                dict_name = f'{ep_len}EL_' + chkpt_name + '_' + f'{start_date.strftime("%b %d %Y")}' +\
                            '_to_' + f'{end_date.strftime("%b %d %Y")}'

                try:
                    with open('./env_results/' + args.setting + '/' + dict_name + '.pkl', 'rb') as file:
                        results = pickle.load(file)

                    for key in results.keys():
                        # Calculation of MSE for only P concentration (Objective)
                        p_mse = mean_squared_error(
                            results[key]['y_real'][:, -1], results[key]['y_pred'][:, -1])

                        df_data.loc[key, checkpoint + '_mse'] = p_mse
                        df_data.loc[key, checkpoint +
                                    '_dtw'] = results[key]['loss_dtw']

                        if loss_type == 'mse':
                            df_loss.loc[checkpoint, key] = p_mse
                            # df_loss.loc[checkpoint, key] = results[key]['test_loss']
                        elif loss_type == 'dtw':
                            df_loss.loc[checkpoint,
                                        key] = results[key]['loss_dtw']
                        elif loss_type == 'both':
                            df_mse.loc[checkpoint,
                                       key] = results[key]['test_loss']
                            df_dtw.loc[checkpoint,
                                       key] = results[key]['loss_dtw']

                except FileNotFoundError:
                    print(f'File for {dict_name} not found.')

        add_experiments()

        def find_best(df_loss, df_mse, df_dtw):
            if loss_type == 'both':
                scaler = MinMaxScaler()

                def resample_df(df):
                    df.columns = pd.to_datetime(df.columns)
                    df['loss_average'] = df.mean(axis=1)
                    df['loss_average'] = df['loss_average'].astype(float)
                    df.columns = df.columns.astype(str)
                    df.iloc[:, :] = scaler.fit_transform(df.iloc[:, :])
                    for column in df.columns:
                        if column != 'loss_average':
                            df.rename(
                                columns={column: pd.to_datetime(column)}, inplace=True)

                    return df

                df_mse = resample_df(df_mse)
                df_dtw = resample_df(df_dtw)
                df_loss.columns = pd.to_datetime(df_loss.columns)
                df_loss['loss_average'] = None

                for i in range(len(df_loss)):
                    for col in df_loss.columns:
                        df_loss[col].iloc[i] = alpha * df_mse[col].iloc[i] + \
                            (1 - alpha) * df_dtw[col].iloc[i]

                df_loss = df_loss.astype(float)
                df_loss_copy = copy.deepcopy(df_loss)
                df_loss_copy = df_loss_copy.drop(columns=['loss_average'])
                df_loss_copy.columns = pd.to_datetime(df_loss_copy.columns)
                df_monthly = df_loss_copy.resample('M', axis=1).mean()
                df_monthly.columns = df_monthly.columns.strftime('%b')
                df_monthly['loss_average'] = df_loss['loss_average']
                df_monthly.sort_values('loss_average')
                df_loss = df_loss.drop('loss_average', axis=1)

            else:
                df_loss.columns = pd.to_datetime(df_loss.columns)
                # print(df_loss['loss_average'].nsmallest(5).index)
                df_monthly = df_loss.resample('M', axis=1).mean()
                df_monthly.columns = df_monthly.columns.strftime('%b')
                df_monthly['loss_average'] = df_monthly.mean(axis=1)
                df_monthly['loss_average'] = df_monthly['loss_average'].astype(
                    float)
                df_monthly.sort_values('loss_average')

            return df_monthly, df_loss, df_mse, df_dtw

        df_monthly, df_loss, df_mse, df_dtw = find_best(
            df_loss, df_mse, df_dtw)

        # Edit df_data
        df_data.index = pd.to_datetime(df_data.index)
        df_data = df_data.resample('M', axis=0).mean()
        for experiment in best_retrained.keys():
            exp_dict = {key: None for key in ['name', 'loss', 'mse', 'dtw']}
            items = list([x for x in df_monthly.index if experiment in x])
            if items != [] and df_monthly.loc[items, 'loss_average'].isna().all() == False:
                all_loss_dict[experiment] = df_monthly.loc[items]
                min_loss_idx = df_monthly.loc[items, 'loss_average'].idxmin()
                exp_dict['name'] = min_loss_idx
                exp_dict['loss'] = df_monthly.loc[min_loss_idx, 'loss_average']
                exp_dict['mse'] = df_data[min_loss_idx + '_mse'].mean()
                exp_dict['dtw'] = df_data[min_loss_idx + '_dtw'].mean()
            else:
                exp_dict['name'] = None
                exp_dict['loss'] = None
                exp_dict['mse'] = None
                exp_dict['dtw'] = None
            best_retrained[experiment] = exp_dict

        return best_retrained, df_monthly, df_loss, all_loss_dict, df_data

    def write_metrics(self, models, mse_dict, dtw_dict):
        df_loss = pd.DataFrame(columns=['points'])
        points_names = [x for x in mse_dict.keys()]
        season_dict = {'Sep': 'Autumn', 'Oct': 'Autumn', 'Nov': 'Autumn',
                       'Dec': 'Winter', 'Jan': 'Winter', 'Feb': 'Winter',
                       'Mar': 'Spring', 'Apr': 'Spring', 'May': 'Spring',
                       'Jun': 'Summer', 'Jul': 'Summer', 'Aug': 'Summer'}

        df_loss['points'] = points_names
        df_loss = df_loss.set_index(['points'])

        for model in models:
            for point_idx, (point) in enumerate(points_names):
                loss_dtw = dtw_dict[point][model]
                mse_list = mse_dict[point][model]
                mse = round(np.mean(mse_list), 3)

                df_loss.loc[point, f'{model}_mse'] = round(mse, 3)
                df_loss.loc[point, f'{model}_dtw'] = round(loss_dtw, 3)

        index_list = df_loss.index.tolist()
        for i, point in enumerate(points_names):
            for key in season_dict.keys():
                if key in point:
                    season = season_dict.get(key)
                    index_list[i] = season

        df_loss.index = index_list
        df_loss.loc['Average', :] = round(df_loss.mean(axis=0), 3)

        return df_loss

    def write_to_tex(self, df, path, name, caption, label, index_name, multi_col=2, highlight=False):
        # Function to style the minimum loss in each row and column
        def highlight_best(data, bold_rows=True, underline_cols=True):
            # Define the styles
            bold = "\\textbf{{{:0.4f}}}"
            underline = "\\underline{{{:0.4f}}}"
            bold_underline = "\\underline{{\\textbf{{{:0.4f}}}}}"
        
            columns = data.columns.to_list()
            # Initialize styles to empty strings
            styles = pd.DataFrame('', index=data.index, columns=columns)
        
            mse_cols = [col for col in columns if 'mse' in col]
            dtw_cols = [col for col in columns if 'dtw' in col]
        
            row_min_indices = set()
            col_min_indices = set()
        
            # Apply styles for minimum in rows
            if bold_rows:
                for idx, row in data[mse_cols].iterrows():
                    if not row.isna().all():
                        min_idx = row.idxmin()
                        styles.loc[idx, min_idx] = bold.format(row[min_idx])
                        row_min_indices.add((idx, min_idx))
                for idx, row in data[dtw_cols].iterrows():
                    if not row.isna().all():
                        min_idx = row.idxmin()
                        styles.loc[idx, min_idx] = bold.format(row[min_idx])
                        row_min_indices.add((idx, min_idx))
        
            # Apply styles for minimum in columns
            if underline_cols:
                for col in columns:
                    min_col = data[col].min()
                    min_idx = data[col].idxmin()
                    if (min_idx, col) in row_min_indices:
                        styles.loc[min_idx, col] = bold_underline.format(min_col)
                    else:
                        styles.loc[min_idx, col] = underline.format(min_col)
                    col_min_indices.add((min_idx, col))
        
            # Replace empty strings with actual numbers
            for idx in data.index:
                for col in data.columns:
                    if styles.loc[idx, col] == '':
                        styles.loc[idx, col] = "{:0.4f}".format(data.loc[idx, col])
        
            return styles


        if highlight and not df.isna().all().all():
            df = highlight_best(df)

        df = df.reset_index()
        avg_col = False
        for col in df.columns:
            if 'Average' in col:
                avg_col = True
                break

        column_format = '@{}c|'
        if avg_col:
            for i in range(len(df.columns) - 1 - multi_col):
                column_format += 'l'

            column_format += '|'

            for j in range(multi_col):
                column_format += 'l'
            column_format += '@{}'
        else:
            for i in range(len(df.columns) - 1):
                column_format += 'l'
            column_format += '@{}'

        if highlight:
            tex_df = df.to_latex(escape=False, index=False, header=True,
                                 caption=caption, label=label, column_format=column_format)
        else:
            tex_df = df.to_latex(escape=False, index=False, header=True,
                                 caption=caption, label=label, column_format=column_format)

        header, rest_of_table = tex_df.split('\\midrule')
        old_headers = header.split('\\toprule')[-1]
        header = header.replace(old_headers, '')
        header = header.replace('\\toprule', '')
        if len(df.columns) > 2:
            header = header.replace('\\begin{table}', '\\begin{table*}')
            # header = header.replace('\\begin{tabular}','\\resizebox{\\textwidth}{!}{\\begin{tabular}')
            rest_of_table = rest_of_table.replace(
                '\\end{table}', '\\end{table*}')
        else:
            header = header.replace(
                '\\begin{tabular}', '\\resizebox{\\columnwidth}{!}{\\begin{tabular}')

        # rest_of_table = rest_of_table.replace('\\end{tabular}','\\end{tabular}}')
        new_header_lines = []
        new_header_lines.append('\\toprule')
        new_header_lines.append(
            f'\multicolumn{{1}}{{c|}}{{\multirow{{2}}{{*}}{{{index_name}}}}}')

        # Assume headers are structured as 'MainHeaderX_SubY'
        # where X is the main header number and Y is the subheader number.
        # set() is not good for wehn we want to preserve the order of cols
        # main_headers = set()
        main_headers = []
        for col in df.columns:
            if col == df.columns[0]:
                continue
            main_header, _ = col.rsplit('_', 1)  # Split on the last underscore
            main_headers.append(main_header)
            # main_headers.add(main_header)

        # Function to remove duplicates while preserving order
        def remove_duplicates(input_list):
            seen = set()
            result = []
            for item in input_list:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        
        main_headers = remove_duplicates(main_headers)
        
        # Sort and process main headers
        for main_header in main_headers:
            new_header_lines.append(
                f'& \\multicolumn{{{multi_col}}}{{c}}{{{main_header}}} ')

        new_header_lines[-1] += '\\\\ \cmidrule(l){2-' + \
            f'{len(df.columns)}' + '} \multicolumn{1}{c|}{} &'

        # Add the subheader line
        subheaders = [col.split('_')[-1].upper()
                      for col in df.columns if col != df.columns[0]]
        new_header_lines.append(' & '.join(subheaders) + ' \\\\')
        new_header = header + '\n'.join(new_header_lines)

        if 'Average' not in main_headers:
            rest_of_table = rest_of_table.replace(
                'Average', '\\midrule Average')

        tex_df = new_header + '\\midrule' + rest_of_table
        
        # To avoid mathematical underscore in Tex 
        # tex_df = tex_df.replace('_', '-')
        
        with open(path + name + '.tex', 'w') as f:
            f.write(tex_df)

    def write_params_to_tex(self, df, path, name, caption, label, index_name, multi_col=2):
        df = df.reset_index()

        column_format = '@{}c|'
        for i in range(len(df.columns) - 1):
            column_format += 'l'
        column_format += '@{}'

        tex_df = df.to_latex(escape=False, index=False, header=True,
                             caption=caption, label=label, column_format=column_format)

        header, rest_of_table = tex_df.split('\\midrule')
        old_headers = header.split('\\toprule')[-1]
        header = header.replace(old_headers, '')
        header = header.replace('\\toprule', '')
        if len(df.columns) > 7:
            header = header.replace('\\begin{table}', '\\begin{table*}')
            # header = header.replace('\\begin{tabular}','\\resizebox{\\textwidth}{!}{\\begin{tabular}')
            rest_of_table = rest_of_table.replace('\\end{table}', '\\end{table*}')
        else:
            header = header.replace(
                '\\begin{tabular}', '\\resizebox{\\columnwidth}{!}{\\begin{tabular}')    
            rest_of_table = rest_of_table.replace('\\end{tabular}', '\\end{tabular}}')
            
        new_header_lines = []
        new_header_lines.append('\\toprule')
        new_header_lines.append(f'\multicolumn{{1}}{{c|}}{{\multirow{{2}}{{*}}{{{index_name}}}}}')

        # Assume headers are structured as 'MainHeaderX_SubY'
        # where X is the main header number and Y is the subheader number.
        main_headers = []
        for col in df.columns:
            if col == 'Models':
                continue
            main_header, _ = col.rsplit('_', 1)  # Split on the last underscore
            if main_header == 'params':
                main_header = 'Parameters'
            # main_headers.add(main_header)
            main_headers.append(main_header)

        # Function to remove duplicates while preserving order
        def remove_duplicates(input_list):
            seen = set()
            result = []
            for item in input_list:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        
        main_headers = remove_duplicates(main_headers)

        # Sort and process main headers
        for main_header in sorted(main_headers):
            new_header_lines.append(
                f'& \\multicolumn{{{int((len(df.columns)-1)/len(main_headers))}}}{{c}}{{{main_header}}} ')

        new_header_lines[-1] += '\\\\ \cmidrule(l){2-' + \
            f'{len(df.columns)}' + '} \multicolumn{1}{c|}{} &'

        # Add the subheader line
        subheaders = [col.split('_')[-1]
                      for col in df.columns if col != 'Models']
        new_header_lines.append(' & '.join(subheaders) + ' \\\\')
        new_header = header + '\n'.join(new_header_lines)

        tex_df = new_header + '\\midrule' + rest_of_table
        
        # To avoid mathematical underscore in Tex 
        # tex_df = tex_df.replace('_', '-')
        
        with open(path + name + '.tex', 'w') as f:
            f.write(tex_df)

    def plot_sim_with_cv(self, plot_dict, plot_points, args, plot_base=True, plot_all=False,
                         start_date='', end_date='', el=1440, plot_el=180, normalize=False, fig_name=''):
        n_features = 5
        features = ['Metal', 'NH4', 'NO3', 'Out P', 'P-amout']
        # Defining the plot
        p_names = [key for key in plot_points.keys()]
        p_dates = [date for date in plot_points.values()]
        chkpt_names = []
        for key in plot_dict.keys():
            chkpt_names.append(plot_dict[key]['name'])

        # Defining the plot
        if plot_el > 360:
            columns = 1
            rows = len(p_dates)
            wspace = 0.3
            hspace = 0.5
            figsize = (6, 1.5*rows)
            outer = gridspec.GridSpec(
                rows, columns, wspace=wspace, hspace=hspace)
        else:
            if len(p_dates) < 3:
                columns = len(p_dates)
                rows = 1
                figsize = (7, 4*rows)
            else:
                columns = 2
                rows = len(
                    p_dates)//2 if len(p_dates) % 2 == 0 else len(p_dates)//2 + 1
                figsize = (self.fig_width, 2.2*rows)

            wspace = 0.5
            hspace = 0.3
            outer = gridspec.GridSpec(
                rows, columns, wspace=wspace, hspace=hspace)

        # fig, axs = plt.subplots(rows, columns, figsize=figsize, dpi=self.fig_dpi)
        fig = plt.figure(figsize=figsize, dpi=self.fig_dpi)
        # fig.tight_layout(pad=4.0)

        RESULTS_PATH = './env_results/' + args.setting + '/'

        def load_results(args, point, chkpt_names, normalize):
            results_dict = {key: None for key in chkpt_names}
            point = f'{point.strftime("%b %d %Y")}'
            for chkpt_name in chkpt_names:
                chckpt_dict = {'y_pred': None, 'y_real': None, 'loss': None}
                if chkpt_name == 'Base Model':
                    name = 'Base'
                else:
                    name = chkpt_name
                dict_name = f'{el}EL_' + name + '_' + f'{start_date.strftime("%b %d %Y")}' +\
                            '_to_' + f'{end_date.strftime("%b %d %Y")}'
                try:
                    with open('./env_results/' + args.setting + '/' + dict_name + '.pkl', 'rb') as file:
                        results = pickle.load(file)
                except FileNotFoundError:
                    print(f'File for {dict_name} not found.')
                    pass

                if normalize:
                    scaler = StandardScaler()
                    results[point]['y_real'] = scaler.fit_transform(
                        results[point]['y_real'])
                    results[point]['y_pred'] = scaler.transform(
                        results[point]['y_pred'])

                chckpt_dict['y_pred'] = results[point]['y_pred']
                chckpt_dict['y_real'] = results[point]['y_real']
                chckpt_dict['loss'] = results[point]['test_loss']
                results_dict[chkpt_name] = chckpt_dict

            return results_dict

        for i in range(len(p_dates)):
            point = p_dates[i]
            results_dict = load_results(args, point, chkpt_names, normalize)
            dates = pd.date_range(start=pd.to_datetime(
                point), end=pd.to_datetime(point) + plot_el*self.freq, freq=self.freq)
            dates = pd.to_datetime(dates)
            times = dates.strftime('%H:%M')

            # Defining the inner and set the title
            if plot_all:
                inner = gridspec.GridSpecFromSubplotSpec(
                    n_features, 1, subplot_spec=outer[i], wspace=0.1, hspace=0, height_ratios=[2, 1, 1, 1, 1])
                ax_0 = plt.Subplot(fig, inner[0])
                inner_axs = []
                for n_plot in range(n_features-1):
                    ax = plt.Subplot(fig, inner[n_plot+1], sharex=ax_0)
                    fig.add_subplot(ax)
                    inner_axs.append(ax)
            else:
                inner = gridspec.GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=outer[i], wspace=0.1, hspace=0, height_ratios=[2, 1])
                ax_0 = plt.Subplot(fig, inner[0])
                inner_axs = []
                ax_1 = plt.Subplot(fig, inner[1], sharex=ax_0)
                fig.add_subplot(ax_1)
                inner_axs.append(ax_1)

            fig.add_subplot(ax_0)
            ax_0.set_title(f'{p_names[i]}')
            # ax.set_title(f'{p_names[i]}: {point.strftime("%b %d %Y")}')

            for j, item in enumerate(plot_dict.keys()):
                name = plot_dict[item]['name']
                plot_legend = f'{item}'
                if item == 'Base Model':
                    ax2 = ax_0.twinx()
                    ax2.plot(results_dict[name]['y_pred'][:plot_el, -1], 'x-',
                             color='purple', label=plot_legend, linewidth=0.5, markersize=0.2)
                else:
                    ax_0.plot(results_dict[name]['y_pred'][:plot_el, -1],
                              'x-', label=plot_legend, linewidth=0.5, markersize=0.2)
                if j == len(plot_dict.keys()) - 1:
                    ax_0.plot(results_dict[name]['y_real'][:plot_el, -1], 'x-',
                              label='Actual-P', linewidth=0.5, markersize=0.2, color='black')

                if plot_all:
                    for n_feature, inner_ax in enumerate(inner_axs):
                        inner_ax.plot(results_dict[name]['y_real'][:plot_el, n_feature],
                                      'x-', linewidth=0.25, markersize=0.1, color='brown')
                else:
                    ax_1.plot(results_dict[name]['y_real'][:plot_el, 0],
                              'x-', linewidth=0.25, markersize=0.1, color='brown')

            if plot_all:
                ax_1 = inner_axs[-1]

            for ax in inner_axs:
                ax.grid(visible=True, which='major',
                        color='gray', linewidth=0.075)
                ax.grid(visible=True, which='minor',
                        color='gray', linewidth=0.075)

            ax_1.set_xticks(np.arange(len(times)))
            ax_1.set_xticklabels(times)
            ax_1.xaxis.set_major_locator(MultipleLocator(int(plot_el/6)))
            ax_1.xaxis.set_minor_locator(MultipleLocator(10))
            ax_1.tick_params(axis='both', which='both', labelsize=6,
                             labelbottom=True, labelleft=True, length=0)

            ax_0.tick_params(axis="y", labelsize=6)
            ax_0.tick_params(which='both', length=0)
            ax_0.grid(visible=True, which='major',
                      color='gray', linewidth=0.075)
            ax_0.grid(visible=True, which='minor',
                      color='gray', linewidth=0.075)

            ax2.tick_params(axis='y', labelcolor='black')
            ax2.tick_params(axis="y", labelsize=6)
            ax2.tick_params(axis=u'both', which=u'both', length=0)

            if i == ((rows-1)*columns):
                if plot_all:
                    for i, ax in enumerate(inner_axs):
                        ax.set_ylabel(features[i], labelpad=5, fontsize=6)
                        if i != len(inner_axs)-1:
                            ax.set_xticks([])
                            ax.set_xticklabels([])
                            ax.tick_params(axis='y', labelsize=6)
                    ax_1 = inner_axs[-1]
                else:
                    ax_1.set_ylabel(
                        'Metal Dosage [m^3/hr]', labelpad=5, fontsize=6)

                ax_0.set_ylabel('P-amount [mg/L]', labelpad=5, fontsize=6)
                ax_1.set_xlabel('Time (24-hour)', labelpad=5, fontsize=6)
                ax2.set_ylabel('Base Model', labelpad=5, fontsize=6)

            handles, labels = ax2.get_legend_handles_labels()
            handles2, labels2 = ax_0.get_legend_handles_labels()

            plt.setp(ax_1.get_xticklabels(), visible=True)
            plt.setp(ax_0.get_xticklabels(), visible=False)

        # for i in range(outer_columns):
           # plt.subplot(outer[-1, i]).set_xlabel('Time', labelpad=10)

        # Showing the plot
        # plt.ion()
        # plt.show()
        fig.legend(handles + handles2, labels + labels2, loc='upper center',
                   ncol=6, labelspacing=0., bbox_to_anchor=(0.5, 0.98), fontsize=self.legend_size)
        # fig.subplots_adjust(top=0.8, bottom=0.2, right=0.9, left=0.1)

        # Saving the plot
        if normalize:
            fig_name = 'Normalized_' + fig_name

        if plot_all:
            fig_name = 'ALL_' + fig_name

        plt.savefig(RESULTS_PATH + fig_name + '.svg')
        plt.savefig(RESULTS_PATH + fig_name + '.pdf')
        plt.savefig(RESULTS_PATH + fig_name + '.png')

        plt.close('all')
        
    def plot_mse_single(self, args, results_dict, setting, model, chkpt_name,
                        points_names, ep_len, start_date, end_date):
        y_label = 'Mean Squared Error'
        losses = []
        for key in results_dict.keys():
            losses.append(results_dict[key]['test_loss'])

        fig = plt.figure(figsize=(12, 8), dpi=self.fig_dpi)
        plt.plot(losses, 'x-', color='purple',
                 label='mse', linewidth=0.5, markersize=2)

        plt.xticks(np.arange(len(points_names)), points_names)
        monthly_locator = mdates.MonthLocator()
        plt.gca().xaxis.set_major_locator(monthly_locator)
        plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
        plt.gca().tick_params(which='minor', length=2)

        plt.grid(visible=True, which='major', color='gray', linewidth=0.075)
        plt.grid(visible=True, which='minor', color='gray', linewidth=0.075)

        plt.ylabel(y_label)
        plt.xlabel('Points')

        fig_folder_path = './env_results/' + args.setting + '/'
        if not os.path.exists(fig_folder_path):
            os.makedirs(fig_folder_path)

        fig_name = f'mse_{ep_len}EL_' + chkpt_name + '_' +\
            f'{start_date.strftime("%b %d %Y")}' + \
            '_to_' + f'{end_date.strftime("%b %d %Y")}'
        plt.savefig(fig_folder_path + fig_name + '.svg')
        plt.savefig(fig_folder_path + fig_name + '.pdf')
        plt.savefig(fig_folder_path + fig_name + '.png')

        # plt.ion()
        # plt.show()
        plt.close('all')
        print(f'The figure for {chkpt_name} is produced.')

    def extract_model_info(self, model_folder):
            """
            Extract dataset name, in features, and out features from the model folder name.
            """
            parts = model_folder.split('_')
            data_tag = parts[1]
            in_features = parts[3].replace('F', '')
            out_features = parts[4].replace('Out', '')
            return data_tag, in_features, out_features   
        
    def plot_mse_together(self, args, plot_dict, df_plot, plot_base, points_names,
                          start_date, end_date, loss_type='mse', alpha=0.5, ep_len=1440, min_max=False,
                          twin=False, log=True, all_best=False, all_best_name='All', plot_type='best_by_ex',
                          do_custom_names=False, custom_model_names=None):

        fig_name = f'mse_all_{ep_len}EL_' + f'{start_date.strftime("%b %d %Y")}' +\
            f'_to_{end_date.strftime("%b %d %Y")}'

        if plot_base == False:
            fig_name = f'mse_all_no_base_{ep_len}EL_' + f'{start_date.strftime("%b %d %Y")}' +\
                f'_to_{end_date.strftime("%b %d %Y")}'

        fig, ax = plt.subplots(figsize=(6.2, 3.5), dpi=self.fig_dpi)
        if loss_type == 'mse':
            y_label = 'Mean Squared Error'
        elif loss_type == 'dtw':
            y_label = 'Dynamic Time Warping'
        else:
            y_label = f'{alpha:0.1f} * MSE + {(1-alpha):0.1f} * DTW'
        x_label = 'Date'
        if log:
            ax.set_yscale('log')
        
        for i, item in enumerate(plot_dict.keys()):
            if all_best:
                if plot_type == 'best_by_ex':
                    plot_legend = f'{item} - {plot_dict[item]["name"]}'
                    name = plot_legend
                elif plot_type == 'best_by_model':
                    if do_custom_names:
                        plot_legend = custom_model_names[item]
                    else:
                        data_tag, in_features, out_features = self.extract_model_info(item)
                        plot_legend = f'{data_tag}-{in_features}i{out_features}o'
                    name = item
            else:
                if plot_type == 'best_by_ex':
                    plot_legend = f'{item}'
                    # plot_legend = f'{item} [Avg.: {plot_dict[item]["info"]["loss"]:.4f}]'
                    # name = plot_dict[item]["name"]
                    name = f'{item} - {plot_dict[item]["name"]}'
                elif plot_type == 'best_by_model':
                    if do_custom_names:
                        plot_legend = custom_model_names[item]
                    else:
                        data_tag, in_features, out_features = self.extract_model_info(item)
                        plot_legend = f'{data_tag}-{in_features}i{out_features}o'
                    name = item
            
            # plot_legend = plot_legend.split('_')[0]
            if item == 'Base Model':
                if plot_base:
                    if twin:
                        ax2 = ax.twinx()
                        ax2.plot(df_plot[name], 'x-', color='purple',
                                 label=plot_legend, linewidth=0.5, markersize=1)
                        ax2.tick_params(axis='y', labelcolor='black')
                        ax2.set_ylabel(y_label + ' (Base Model)',
                                       labelpad=10, fontsize=10)
                    elif log:
                        ax.plot(df_plot[name], 'x-', color='purple',
                                label=plot_legend, linewidth=0.5, markersize=1)

                else:
                    continue
            else:
                ax.plot(df_plot[name], 'x-', label=plot_legend,
                        linewidth=0.5, markersize=1)

        ax.tick_params(axis='y', labelcolor='black', labelsize=10)
        freq_add = datetime.timedelta(days=1)
        points_names.append(
            f'{(pd.to_datetime(points_names[-1]) + freq_add).strftime("%b %d %Y")}')

        if min_max:
            max_mse = None
            min_mse = None
            df_plot['point_avg'] = df_plot.mean(axis=1)
            max_mse = df_plot['point_avg'].max()
            max_mse_idx = df_plot['point_avg'].idxmax()
            min_mse = df_plot['point_avg'].min()
            min_loss_idx = df_plot['point_avg'].idxmin()
            ax.annotate(f'Max: {max_mse_idx.strftime("%b %d %Y")}', xy=(max_mse_idx, max_mse),
                        xytext=(max_mse_idx - freq_add*80, max_mse - 0.01),
                        arrowprops=dict(arrowstyle='->', color='black'), fontsize=8)

            ax.annotate(f'Min: {min_loss_idx.strftime("%b %d %Y")}', xy=(min_loss_idx, min_mse),
                        xytext=(min_loss_idx - freq_add*80, min_mse + 0.025),
                        arrowprops=dict(arrowstyle='->', color='black'), fontsize=8)

        month_day_formatter = mdates.DateFormatter('%b %y')
        ax.xaxis.set_major_formatter(month_day_formatter)
        day_locator = mdates.DayLocator(interval=5)
        ax.xaxis.set_minor_locator(day_locator)
        ax.tick_params(which='minor', length=2, labelsize=10)
        ax.tick_params(axis='x', labelsize=10)
        # ax.tick_params(axis="x", rotation=45)
        # plt.xticks(rotation=45)

        plt.grid(visible=True, which='major',
                 color='lightgray', linewidth=0.0025)
        plt.grid(visible=True, which='minor',
                 color='lightgray', linewidth=0.0025)

        ax.set_ylabel(y_label, labelpad=6, fontsize=10)
        ax.set_xlabel(x_label, labelpad=6, fontsize=10)

        # plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.9)
        lines, labels = ax.get_legend_handles_labels()
        if plot_base:
            if twin:
                lines2, labels2 = ax2.get_legend_handles_labels()
                fig.legend(lines + lines2, labels + labels2, loc='upper center', ncol=5,
                           labelspacing=0., bbox_to_anchor=(0.5, 0.98), fontsize=self.legend_size)

            fig.legend(lines, labels, loc='upper center', ncol=5,
                       labelspacing=0.2, bbox_to_anchor=(0.5, 0.98), fontsize=self.legend_size)
        else:
            fig.legend(lines, labels, loc='upper center', ncol=2,
                       labelspacing=0.2, bbox_to_anchor=(0.5, 0.98), fontsize=self.legend_size)

        if all_best:
            fig_folder_path = './env_results/All Best/'
            fig_name = f'{all_best_name}_loss_{ep_len}EL_{start_date.strftime("%b %d %Y")}' +\
                f'_to_{end_date.strftime("%b %d %Y")}'
        else:
            fig_folder_path = './env_results/' + args.setting + '/'

        if not os.path.exists(fig_folder_path):
            os.makedirs(fig_folder_path)

        plt.savefig(fig_folder_path + fig_name + '.svg')
        plt.savefig(fig_folder_path + fig_name + '.pdf')
        plt.savefig(fig_folder_path + fig_name + '.png')

        plt.ion()
        # plt.show()
        plt.close('all')

    def plot_box_heat(self, args, ex_list, df_plot, ep_len, start_date, end_date,
                      plot_base=False, loss_type='mse', alpha=0.5, all_best=False, all_best_name='All'):

        df_plot_m = df_plot.resample('M', axis=0).mean()
        df_plot_m.index = df_plot_m.index.strftime('%b')
        if loss_type == 'mse':
            label = 'Average Mean Squared Error'
        elif loss_type == 'dtw':
            label = 'Average Dynamic Time Warping'
        else:
            label = f'Average {alpha:0.1f} * MSE + {(1-alpha):0.1f} * DTW'
        for experiment in ex_list:
            for col in df_plot_m.columns:
                if experiment in col:
                    df_plot_m.rename(columns={col: experiment}, inplace=True)

        def plot_heat(df, label):
            fig_name = f'heatmap_base_{ep_len}EL_' + f'{start_date.strftime("%b %d %Y")}' +\
                '_to_' + f'{end_date.strftime("%b %d %Y")}' + '_hor'
            
            fig_height = 0.7
            fig_width = self.fig_width
            if plot_base == False:
                df = df.drop('Base Model', axis=1)
                fig_name = f'heatmap_no_base_{ep_len}EL_' + f'{start_date.strftime("%b %d %Y")}' +\
                    '_to_' + f'{end_date.strftime("%b %d %Y")}' + '_hor'
                    
                fig_height = 2.5

            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=self.fig_dpi)
            # sns.set(font_scale=0.75)
            ax = sns.heatmap(df.round(3).T,
                             cbar=False,
                             cbar_kws={'label': label,
                                       'orientation': 'horizontal'},
                             annot=True,
                             square=True,
                             linewidths=0.75, cmap="YlGnBu",
                             fmt=".3f",
                             annot_kws={"size": 10})

            # ax.set(ylabel=None)
            # ax.xaxis.tick_bottom()
            # ax.set_xlabel('Months', labelpad=10)
            # ax.set_ylabel('Months', labelpad=8)
            ax.set(xlabel=None)
            ax.set(ylabel=None)
            if plot_base:
                ax.set_xticklabels([])
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
            else:
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
            
            cbar = ax.collections[0].colorbar
            if cbar != None:
                cbar.set_label(label=label, fontsize=10)
                cbar.ax.tick_params(labelsize=10)

            # ver
            # plt.subplots_adjust(left=0.0, right=0.85, bottom=0.08, top=0.98)
            # hor
            if plot_base:
                plt.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
            else:
                plt.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.98)
            # plt.tight_layout()
            if all_best:
                fig_folder_path = './env_results/All Best/'
                fig_name = f'{all_best_name}_heatmap_{ep_len}EL_{start_date.strftime("%b %d %Y")}' +\
                    f'_to_{end_date.strftime("%b %d %Y")}'
            else:
                fig_folder_path = './env_results/' + args.setting + '/'
            if not os.path.exists(fig_folder_path):
                os.makedirs(fig_folder_path)

            plt.savefig(fig_folder_path + fig_name + '.svg')
            plt.savefig(fig_folder_path + fig_name + '.pdf')
            plt.savefig(fig_folder_path + fig_name + '.png')

            plt.close('all')

        def plot_box(df, label):
            fig_name = f'box_{ep_len}EL_' + f'{start_date.strftime("%b %d %Y")}' +\
                '_to_' + f'{end_date.strftime("%b %d %Y")}'

            if plot_base == False:
                df = df.drop('Base Model', axis=1)
                fig_name = f'box_no_base_{ep_len}EL_' + f'{start_date.strftime("%b %d %Y")}' +\
                    '_to_' + f'{end_date.strftime("%b %d %Y")}'

            fig, ax = plt.subplots(figsize=(5, 7.4), dpi=self.fig_dpi)
            # sns.set(font_scale=0.75)
            ax = sns.boxplot(df.round(5), palette='coolwarm')
            ax.xaxis.tick_bottom()

            ax.set_xlabel('Models', labelpad=10)
            ax.set_ylabel(label, labelpad=10)

            # plt.subplots_adjust(left=0.12, right=0.95, bottom=0.25, top=0.95)
            plt.tight_layout()
            if all_best:
                fig_folder_path = './env_results/All Best/'
                fig_name = f'{all_best_name}_box_{ep_len}EL_{start_date.strftime("%b %d %Y")}' +\
                    f'_to_{end_date.strftime("%b %d %Y")}'
            else:
                fig_folder_path = './env_results/' + args.setting + '/'

            if not os.path.exists(fig_folder_path):
                os.makedirs(fig_folder_path)

            plt.savefig(fig_folder_path + fig_name + '.svg')
            plt.savefig(fig_folder_path + fig_name + '.pdf')
            plt.savefig(fig_folder_path + fig_name + '.png')

            plt.close('all')

        # df_plot_m = df_plot_m.drop(columns='point_avg')
        plot_heat(df_plot_m, label)
        plot_box(df_plot_m, label)

    def plot_retrained(self, plot_dict, plot_points, args=None, plot_base=True,
                       start_date='', end_date='', el=1440, plot_el=180, normalize=False, fig_name='',
                       all_best=False, all_best_name='All', plot_type_best='best_by_ex', latex_path='', seq_len=240,
                       do_custom_names=False, custom_model_names=None):
        # print(plot_dict)
        # model_name = 'LSTM'
        p_names = [key for key in plot_points.keys()]
        p_dates = [date for date in plot_points.values()]
        chkpt_names = []
        model_names = []
        for key in plot_dict.keys():
            if all_best:
                if plot_type_best == 'best_by_ex':
                    model_names.append(plot_dict[key]['info']['model_name'])
                    chkpt_names.append(plot_dict[key]['info']['ex_name'])
                elif plot_type_best == 'best_by_model':
                    model_names.append(key)
                    chkpt_names.append(plot_dict[key]['ex_name'])
            else:
                if plot_type_best == 'best_by_ex':
                    chkpt_names.append(plot_dict[key]['info']['ex_name'])
                elif plot_type_best == 'best_by_model':
                    chkpt_names.append(plot_dict[key]['ex_name'])

        # Defining the plot
        if plot_el > 360:
            columns = 1
            rows = len(p_dates)
            wspace = 0.3
            hspace = 0.5
            figsize = (6, 1.5*rows)
        else:
            if len(p_dates) < 3:
                columns = len(p_dates)
                rows = 1
                figsize = (7, 4*rows)
            else:
                columns = 2
                rows = len(
                    p_dates)//2 if len(p_dates) % 2 == 0 else len(p_dates)//2 + 1
                figsize = (self.fig_width, 2.2*rows)

            wspace = 0.5
            hspace = 0.3

        fig, axs = plt.subplots(
            rows, columns, figsize=figsize, dpi=self.fig_dpi)
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        # padding of the figure
        # fig.tight_layout(pad=4.0)
        if all_best:
            RESULTS_PATH = './env_results/All Best/'
        else:
            RESULTS_PATH = './env_results/' + args.setting + '/'
        # Plotting

        def load_results(point, chkpt_names, model_names=[], args=None, normalize=False):
            results_dict = {key: None for key in chkpt_names}
            point = f'{point.strftime("%b %d %Y")}'
            for i, chkpt_name in enumerate(chkpt_names):
                chckpt_dict = {'y_pred': None, 'y_real': None, 'loss': None}
                if chkpt_name == 'Base Model':
                    if all_best:
                        name = 'Base'
                    else:
                        name = ''
                else:
                    name = chkpt_name
                dict_name = f'{el}EL_' + name + '_' + f'{start_date.strftime("%b %d %Y")}' +\
                            '_to_' + f'{end_date.strftime("%b %d %Y")}'
                try:
                    if all_best:
                        model_name = model_names[i]
                        path = './env_results/' + model_name + '/' + dict_name + '.pkl'
                    else:
                        path = './env_results/' + args.setting + '/' + dict_name + '.pkl'

                    with open(path, 'rb') as file:
                        results = pickle.load(file)
                except FileNotFoundError:
                    print(f'File for {dict_name} not found.')
                    pass

                if normalize:
                    scaler = StandardScaler()
                    results[point]['y_real'] = scaler.fit_transform(
                        results[point]['y_real'])
                    results[point]['y_pred'] = scaler.transform(
                        results[point]['y_pred'])

                chckpt_dict['y_pred'] = results[point]['y_pred']
                chckpt_dict['y_real'] = results[point]['y_real']
                chckpt_dict['loss'] = results[point]['test_loss']
                results_dict[chkpt_name] = chckpt_dict

            return results_dict

        mse_dict = {key: None for key in p_names}
        dtw_dict = {key: None for key in p_names}
        for (i, ax) in zip(range(len(p_dates)), axs.ravel()):
            point = p_dates[i]
            global results_dict
            results_dict = load_results(
                    point, chkpt_names, model_names=model_names, args=args, normalize=normalize)

            dates = pd.date_range(start=pd.to_datetime(
                point), end=pd.to_datetime(point) + plot_el*self.freq, freq=self.freq)
            dates = pd.to_datetime(dates)
            times = dates.strftime('%H:%M')
            ax.set_title(f'{p_names[i]}')
            # ax.set_title(f'{p_names[i]}: {point.strftime("%b %d %Y")}')

            mse_point = {key: None for key in plot_dict.keys()}
            dtw_point = {key: None for key in plot_dict.keys()}
            for j, item in enumerate(plot_dict.keys()):
                if plot_type_best == 'best_by_ex':
                    name = plot_dict[item]['info']['ex_name']
                    # name = plot_dict[item]['info']['ex_name'] if all_best else plot_dict[item]['name']
                    plot_legend = f'{item} - {plot_dict[item]["name"]}' if all_best else f'{item}'
                    
                    plot_legend = plot_legend.split('_')[0]
                elif plot_type_best == 'best_by_model':
                    name = plot_dict[item]['ex_name']
                    if do_custom_names:
                        plot_legend = custom_model_names[item]
                    else:
                        data_tag, in_features, out_features = self.extract_model_info(item)
                        plot_legend = f'{data_tag}-{in_features}i{out_features}o'
                
                y_real = results_dict[name]['y_real'][:plot_el, -1]
                y_pred = results_dict[name]['y_pred'][:plot_el, -1]

                mse_list = []
                for step in range(len(y_real)):
                    mse = mean_squared_error(y_real, y_pred)
                    mse_list.append(mse)

                # mse_list = (mse_list-np.min(mse_list))/(np.max(mse_list)-np.min(mse_list))
                mse_point[item] = mse_list

                loss_dtw, _ = self.compute_dtw(y_real, y_pred)
                dtw_point[item] = loss_dtw
                
                ax2 = None
                if item == 'Base Model':
                    ax2 = ax.twinx()
                    ax2.plot(y_pred, 'x-', color='purple',
                             label=plot_legend, linewidth=0.5, markersize=0.2)
                    ax2.tick_params(axis='y', labelcolor='black')
                    ax2.set_ylabel('Base Model', labelpad=5)
                else:
                    ax.plot(y_pred, 'x-', label=plot_legend,
                            linewidth=0.5, markersize=0.2)
                if j == len(plot_dict.keys()) - 1:
                    ax.plot(y_real, 'x-', label='Actual-P',
                            linewidth=0.5, markersize=0.2, color='black')

            mse_dict[p_names[i]] = mse_point
            dtw_dict[p_names[i]] = dtw_point

            ax.set_xticks(np.arange(len(times)))
            ax.set_xticklabels(times)
            ax.xaxis.set_major_locator(MultipleLocator(plot_el/6))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.tick_params(which='minor', length=2)

            ax.set_ylabel('P-amount [mg/L]', labelpad=5, fontsize=8)
            if ax2 is not None:
                handles, labels = ax2.get_legend_handles_labels()
            handles2, labels2 = ax.get_legend_handles_labels()
            
            # Add a rectangle
            # Define the rectangle properties
            x_start = 0  # Start from the first time point
            x_end = seq_len  # End at the specified time point
            width = (x_end - x_start)  # Convert to days for matplotlib
            
            def solid_color():
                # Add the rectangle patch
                rect = patches.Rectangle((x_start, 0), width, 1, transform=ax.get_xaxis_transform(),
                                         linewidth=1, edgecolor='none', facecolor='skyblue', alpha=0.5)
                ax.add_patch(rect)
            
            def gradient_color():
                # Define the gradient colors
                cmap = LinearSegmentedColormap.from_list('fade', ['skyblue', 'white'])
                
                # Number of gradient steps
                n_steps = int(seq_len * (self.freq.total_seconds()/60))
                
                # Create gradient rectangles
                verts = []
                colors = []
                
                for i in range(n_steps):
                    x0 = x_start + i * (x_end - x_start) / n_steps
                    x1 = x_start + (i + 1) * (x_end - x_start) / n_steps
                    verts.append([(x0, 0), (x0, 1), (x1, 1), (x1, 0)])
                    colors.append(cmap(i / n_steps))
                
                # Create a PolyCollection
                gradient = PolyCollection(verts, facecolors=colors, edgecolor='none', transform=ax.get_xaxis_transform(), zorder=1)
                
                # Add the gradient to the plot
                ax.add_collection(gradient)
            
            # solid_color()
            gradient_color()
            
            ax.grid(visible=True, which='major', color='silver', linewidth=0.025, zorder=5)
            ax.grid(visible=True, which='minor', color='silver', linewidth=0.025, zorder=5)
            ax.set_axisbelow(False)  # Ensure gridlines are above other elements
            
            ax.margins(0.005)           # Default margin is 0.05, value 0 means fit
            
            ax.yaxis.set_label_coords(-0.05, 0.5)   # Aligning the y labels

        for a in range(columns):
            if len(p_dates) < columns + 1:
                axs[a].set_xlabel('Time (24-hour)', labelpad=10, fontsize=8)
            elif columns == 1 and rows > 1:
                axs[-1].set_xlabel('Time (24-hour)', labelpad=10, fontsize=8)
            else:
                axs[-1, a].set_xlabel('Time (24-hour)',
                                      labelpad=10, fontsize=8)

        # for i in range(outer_columns):
           # plt.subplot(outer[-1, i]).set_xlabel('Time', labelpad=10)
        # Showing the plot
        # plt.ion()
        # plt.show()
        if plot_type_best == 'best_by_ex':
            fig.legend(handles + handles2, labels + labels2, loc='upper center',
                       ncol=6, labelspacing=0., bbox_to_anchor=(0.5, 0.98), fontsize=self.legend_size)
        if plot_type_best == 'best_by_model':
            fig.legend(handles2, labels2, loc='upper center',
                       ncol=6, labelspacing=0., bbox_to_anchor=(0.5, 0.98), fontsize=self.legend_size)
            
        fig.subplots_adjust(top=0.9, bottom=0.08, right=0.98, left=0.08)
        
        # Saving the plot
        if normalize:
            fig_name = 'Normalized_' + fig_name
                            
        plt.savefig(RESULTS_PATH + fig_name + '.svg')
        plt.savefig(RESULTS_PATH + fig_name + '.pdf')
        plt.savefig(RESULTS_PATH + fig_name + '.png')

        plt.close('all')
                
        df_loss = self.write_metrics(
            [x for x in plot_dict.keys()], mse_dict, dtw_dict)
        
        if plot_type_best == 'best_by_model':                
            new_columns = []
            for col in df_loss.columns:
                if do_custom_names and custom_model_names != None:
                    loss = col.split('_')[-1]
                    new_name = custom_model_names[col.replace(f'_{loss}','')]
                    new_name = new_name + f'_{loss}'
                else:
                    data_tag, in_features, out_features = self.extract_model_info(col)
                    loss = col.split('_')[-1]
                    new_name = f'{data_tag}-{in_features}i{out_features}o_{loss}'
                new_columns.append(new_name)
            df_loss.columns = new_columns
                
        df_loss.to_csv(RESULTS_PATH + f'Loss_{fig_name}.csv', index=True)

        fig_name_parts = fig_name.split('_')
        name = 'metrics_table_' + fig_name_parts[0] + '_' + fig_name_parts[2] + \
            '_' + fig_name_parts[3] + '_' + fig_name_parts[4] + '_' + plot_type_best
        
        if  plot_type_best == 'best_by_ex':
            caption = 'The average Mean Squared Error and Dynamic Time Warping data for each model in the different points of the year.'+\
                ' The best MSE and DTW values for each point and model are highlighted in bold and underlined, respectively.'
        else:
            caption = 'The average Mean Squared Error and Dynamic Time Warping data for the base model and improved versions' +\
            ' during different seasons of the year. The best values of MSE and DTW for each month are highlightd in bold.'
        if all_best:
            label = f'tab:{plot_el}_metrics_{all_best_name}_{plot_type_best}'
        else:
            label = f'tab:{plot_el}_metrics_{plot_type_best}'

        index_name = 'Points'
        self.write_to_tex(df_loss, latex_path, name, caption,
                          label, index_name, multi_col=2, highlight=True)
