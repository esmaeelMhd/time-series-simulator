import pandas as pd
import numpy as np
import datetime
import itertools
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

import torch

from utils.env_helper import EnvHelper

#%% Plot and device options

# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.xmargin'] = 0.02
plt.rcParams['axes.ymargin'] = 0.04
fig_dpi = 500
# plt.switch_backend('qt5agg')

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"{device}" " is available.")

# Frequency of the dataset
freq = datetime.timedelta(minutes=1)

#%% Initializing

# The raw dataset
dataset_name = 'wastewater.csv'
plot_mode = 'Together'
mse_all = False
mse_p = True
model_mse_change = 'LSTM'

# If retrained
retrained = False
retrained_checkpoint_name = '120_episodes_12_months_with_gap_act_actions_10Epochs_retrained_checkpoint.pth' 

# Environment and the model specifications    
episode_length = 180
sequence_length = 240

# Create an instance of the helper
helper = EnvHelper()

# Points to test
test_frequency = 'Seasons'      # options: Months, Seasons 
day_of_the_month = 'Middle'     # options: First, Middle  
time_of_the_day = 'Morning'     # options: Morning, Noon
first_date = pd.to_datetime('2021-10')
last_date = pd.to_datetime('2022-07')  
points = test_frequency + '_' + day_of_the_month + '_' + time_of_the_day

list_points, points_names = helper.make_points(test_frequency=test_frequency,
                                               day_of_the_month=day_of_the_month,
                                               time_of_the_day=time_of_the_day,
                                               first_date=first_date,
                                               last_date=last_date)

# Choosing the models to run
single = False
best_test_results = False
best_env_results = True
dataset_based = False
all_models = False
setting = 'LSTM_New_CorrH_11F_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256Hidden_2LayerDim'
data_tag = 'New_CorrH'

models_list, best_models = helper.make_models_list(setting=setting,
                                      single=single,
                                      best_test_results=best_test_results,
                                      best_env_results=best_env_results,
                                      dataset_based=dataset_based,
                                      all_models=all_models,
                                      data_tag=data_tag,
                                      episode_length=episode_length)

df_raw = pd.read_csv('./datasets/'+dataset_name)
df_raw = df_raw.set_index(["date"])
df_raw.index = pd.to_datetime(df_raw.index)
if not df_raw.index.is_monotonic:
    df_raw = df_raw.sort_index()
    
# feature_names = df_raw.columns 
feature_names = list(['T1_PO4'])

print(models_list)

#%% Seperate plots for each point

states_dict = {key: [] for key in best_models}
actual_states_dict = {key: [] for key in best_models}
mse_dict = {key: [] for key in best_models}
    
def plot_separate():
    fig_folder_path = './figures/'
    for point_idx, point_name in enumerate(points_names):
        fig, axs = plt.subplots(3,2, figsize=(16,9), dpi=500)
        fig.tight_layout(pad=4.0)
        fig.suptitle(point_name, fontweight="bold")
        start_date = list_points[point_idx]+freq*sequence_length
        dates = pd.date_range(start=start_date, end=start_date + episode_length*freq, freq='min')
        dates = pd.to_datetime(dates)
        times = dates.strftime('%H:%M')

        colors_dict = {'LSTM': 'blue', 'Transformer': 'orange', 'Informer': 'green',
                      'Autoformer': 'red', 'DLinear': 'purple', 'NLinear': 'brown'}

        for k, (i, j) in enumerate(itertools.product(range(3), range(2))):
            model = list(best_models.keys())[k]
            RESULTS_PATH = './env_results/' + best_models.get(model) + '/'

            states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{point_name}_states.npy')
            real_states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{point_name}_actual_states.npy')

            states = np.array(states).reshape(np.array(states).shape[0], -1)
            real_states = np.array(real_states).reshape(np.array(real_states).shape[0], -1)
            states[:,0] = real_states[:,0]

            mse_list = []
            for step in range(len(states)):
                mse = mean_squared_error(real_states[step,:], states[step,:])
                mse_list.append(mse)

            mse_average = np.mean(mse_list)

            axs[i,j].title.set_text(model + ' | ' + f'mse: {mse_average}')
            axs[i,j].set_ylabel('P-amount [mg/L]', labelpad=10)
            axs[i,j].plot(states[:,-1],  'x-', color=colors_dict[model], label='Simulated-P', linewidth=0.8, markersize=2)
            axs[i,j].plot(real_states[:,-1], color='black', label='Actual-P', linewidth=1.5)
            axs[i,j].grid(visible=True, which='major', color='gray', linewidth=0.075)
            axs[i,j].grid(visible=True, which='minor', color='gray', linewidth=0.075)
            axs[i,j].margins(tight=True)
            axs[i,j].legend()

            axs[i,j].set_xticks(np.arange(len(times)))
            axs[i,j].set_xticklabels(times)
            axs[i,j].xaxis.set_major_locator(MultipleLocator(20))
            axs[i,j].xaxis.set_minor_locator(MultipleLocator(5))
            axs[i,j].tick_params(which='minor', length=2)

            states_dict.setdefault(model, []).append(states)
            actual_states_dict.setdefault(model, []).append(real_states)
            mse_dict.setdefault(model, []).append(mse_list)

        axs[-1,0].set_xlabel('Time', labelpad=10)
        axs[-1,1].set_xlabel('Time', labelpad=10)

        if not os.path.exists(fig_folder_path):    
            os.makedirs(fig_folder_path)
        
        if dataset_based:
            fig_name = f'{point_name}_{episode_length}_rounds'
        else:
            fig_name = f'{point_name}_{episode_length}_rounds'
        
        plt.savefig(fig_folder_path + fig_name + '.svg')
        plt.savefig(fig_folder_path + fig_name + '.pdf')
        plt.savefig(fig_folder_path + fig_name + '.png')

        plt.ion()
        plt.show()

#%%

mse_dict = {key: [] for key in best_models.keys()}
     
def plot_together():
    y_labels = {
        'T1_NH4': 'NH4-amount [mg/L]',
        'T1_NH4NO3': 'NH4NO3-amount [mg/L]',
        'T1_NO3': 'NO3-amount [mg/L]',
        'T1_PO4': 'P-amount [mg/L]',
        'T1_NH4NO3_NCF': 'NH4NO3_NCF-amount []mg/L'
        }
    
    for feature_idx in range(len(feature_names)):
        if 'Seasons' in points:
            fig, axs = plt.subplots(2,2, figsize=(16,9), dpi=500)
        elif 'Months' in points:
            fig, axs = plt.subplots(len(list_points)//2,2, figsize=(20,12), dpi=500)
    
        fig.tight_layout(pad=4.0)
        for i, (point, ax) in enumerate(zip(points_names, axs.ravel())):
            for k, model in enumerate(best_models.keys()):
                RESULTS_PATH = './env_results/' + best_models.get(model) + '/'
    
                if 'Seasons' in points:
                    # file_name = (point.split('|')[1]).lstrip()
                    file_name = point
                    states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{file_name} - {time_of_the_day}_Phosphorous.npy')
                    real_states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{file_name} - {time_of_the_day}_actual_p.npy')
                else:
                    states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{point}_states.npy')
                    real_states = np.load(RESULTS_PATH + f'{episode_length}_rounds_{point}_actual_states.npy')
    
                states = np.array(states).reshape(np.array(states).shape[0], -1)
                real_states = np.array(real_states).reshape(np.array(real_states).shape[0], -1)
                
                states[:,0] = real_states[:,0]
    
                mse_list = []
                for step in range(len(states)):
                    mse = mean_squared_error(real_states[step,:], states[step,:])
                    mse_list.append(mse)
    
                # mse_list = (mse_list-np.min(mse_list))/(np.max(mse_list)-np.min(mse_list))
                mse_average = np.mean(mse_list)
                mse_dict.setdefault(model, []).append(mse_list)
    
                start_date = list_points[i]+freq*sequence_length
                dates = pd.date_range(start=start_date, end=start_date + episode_length*freq, freq='min')
                dates = pd.to_datetime(dates)
                times = dates.strftime('%H:%M')
    
                plot_label = f'{model}'
                y_label = y_labels.get(df_raw.columns[feature_idx])
    

                if k == 0:
                    ax.plot(real_states[:,feature_idx], color='black', label='Actual P', linewidth=1.5)
    
                ax.plot(states[:,feature_idx], 'x-', label=plot_label, linewidth=0.5, markersize=2)
                ax.set_xticks(np.arange(len(times)))
                ax.set_xticklabels(times)
                ax.xaxis.set_major_locator(MultipleLocator(20))
                ax.xaxis.set_minor_locator(MultipleLocator(5))
                ax.tick_params(which='minor', length=2)
    
    
            #ax.set_ylim([-2,7])
            ax.grid(visible=True, which='major', color='gray', linewidth=0.075)
            ax.grid(visible=True, which='minor', color='gray', linewidth=0.075)
        for a in range(axs.shape[1]):   
            axs[-1,a].set_xlabel('Time', labelpad=10)
    
        handles, labels = axs[-1,-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=7, labelspacing=0.)
    
        fig_folder_path = './figures/'
        if not os.path.exists(fig_folder_path):    
            os.makedirs(fig_folder_path)
    
        if dataset_based:  
            fig_name = f'{data_tag}_{feature_names[feature_idx]}_{points}_Together'
        else:
            fig_name = f'{feature_names[feature_idx]}_{points}_Together'
            
        plt.savefig(fig_folder_path + fig_name + '.svg')
        plt.savefig(fig_folder_path + fig_name + '.pdf')
        plt.savefig(fig_folder_path + fig_name + '.png')
    
        # plt.ion()
        # plt.show() 
        plt.close()
        print(f'The figure for {feature_names[feature_idx]} is produced.')
    
def plot_mse():
    if 'Seasons' in points:
        fig_mse, axs_mse = plt.subplots(2,2, figsize=(16,9), dpi=500)
    elif 'Months' in points:
        fig_mse, axs_mse = plt.subplots(len(list_points)//2,2, figsize=(20,12), dpi=500)

    fig_mse.tight_layout(pad=4.0)
    for point_idx, (point, ax_mse) in enumerate(zip(points_names, axs_mse.ravel())):
        for model in best_models.keys():
            mse_list = mse_dict.get(model)[point_idx]

            mse_list = np.array(mse_list).reshape(-1,1)
            scaler = MinMaxScaler()
            mse_list = scaler.fit_transform(mse_list)

            start_date = list_points[point_idx]+freq*sequence_length
            dates = pd.date_range(start=start_date, end=start_date + episode_length*freq, freq='min')
            dates = pd.to_datetime(dates)
            times = dates.strftime('%H:%M')

            plot_label = f'{model}'
            ax_mse.set_title(point, fontweight="bold")
            ax_mse.set_ylabel('Mean Squared Error', labelpad=10)   
            ax_mse.plot(mse_list, 'x-', label=plot_label, linewidth=0.5, markersize=2)
            ax_mse.set_xticks(np.arange(len(times)))
            ax_mse.set_xticklabels(times)
            ax_mse.xaxis.set_major_locator(MultipleLocator(20))
            ax_mse.xaxis.set_minor_locator(MultipleLocator(5))
            ax_mse.tick_params(which='minor', length=2)
        ax_mse.grid(visible=True, which='major', color='gray', linewidth=0.075)
        ax_mse.grid(visible=True, which='minor', color='gray', linewidth=0.075)

    for a in range(axs_mse.shape[1]):   
        axs_mse[-1,a].set_xlabel('Time', labelpad=10)

    handles_mse, labels_mse = axs_mse[-1,-1].get_legend_handles_labels()
    fig_mse.legend(handles_mse, labels_mse, loc='upper center', ncol=7, labelspacing=0.)

    fig_folder_path = './figures/'
    if not os.path.exists(fig_folder_path):    
        os.makedirs(fig_folder_path)
    
    if dataset_based:
        fig_mse_name = f'{data_tag}_mse_{points}_Together'
    else:
        fig_mse_name = f'mse_{points}_Together'
    
    plt.savefig(fig_folder_path + fig_mse_name + '.svg')
    plt.savefig(fig_folder_path + fig_mse_name + '.pdf')
    plt.savefig(fig_folder_path + fig_mse_name + '.png')
    plt.close()

list_change = []
def plot_mse_change():
    if 'Seasons' in points:
        fig_mse_change, axs_mse = plt.subplots(2,2, figsize=(16,9), dpi=500)
    elif 'Months' in points:
        fig_mse_change, axs_mse = plt.subplots(len(list_points)//2,2, figsize=(20,12), dpi=500)

    fig_mse_change.tight_layout(pad=4.0)
    bar_width = 0.4
    for point_idx, (point, ax_mse) in enumerate(zip(points_names, axs_mse.ravel())):
        model = model_mse_change
        mse_list = mse_dict.get(model)[point_idx]

        mse_list = np.array(mse_list).reshape(-1,1)
        scaler = MinMaxScaler()
        mse_list = scaler.fit_transform(mse_list)

        mse_change_df = pd.DataFrame(columns = ['mse', 'mse_change', 'mse_change_first'])
        mse_change_df['mse'] = np.array(mse_list[:,0])
        mse_change_df['mse_change'] = mse_change_df['mse'].pct_change().mul(100)
        list_change.append(mse_change_df['mse_change'])

        mse_change_df['mse_change_first'] = 100 * ((mse_change_df.mse - mse_change_df.iloc[0].mse) / mse_change_df.iloc[0].mse)

        mse_change_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        print(mse_change_df.head())

        start_date = list_points[point_idx]+freq*sequence_length
        dates = pd.date_range(start=start_date, end=start_date + episode_length*freq, freq='min')
        dates = pd.to_datetime(dates)
        times = dates.strftime('%H:%M')

        x = np.arange(episode_length)
        mse_changes = {
            'Change per Step': mse_change_df['mse_change'],
            'Change from the Begining': mse_change_df['mse_change_first']
            }

        for multiplier, (attribute, measurement) in enumerate(mse_changes.items()):
            offset = bar_width * multiplier
            rects = ax_mse.bar(x + offset, measurement, bar_width, label=attribute)
        ax_mse.set_title(point, fontweight="bold")
        ax_mse.set_ylabel('Percentage', labelpad=10)
        ax_mse.set_xticks(np.arange(len(times)) + bar_width)
        ax_mse.set_xticklabels(times)
        ax_mse.xaxis.set_major_locator(MultipleLocator(20))
        ax_mse.xaxis.set_minor_locator(MultipleLocator(5))
        ax_mse.tick_params(which='minor', length=2)

        ax_mse.grid(visible=True, which='major', color='gray', linewidth=0.075)
        ax_mse.grid(visible=True, which='minor', color='gray', linewidth=0.075)

    for a in range(axs_mse.shape[1]):   
        axs_mse[-1,a].set_xlabel('Time', labelpad=10)

    handles_mse, labels_mse = axs_mse[-1,-1].get_legend_handles_labels()
    fig_mse_change.legend(handles_mse, labels_mse, loc='upper center', ncol=7, labelspacing=0.)

    fig_folder_path = './figures/'
    if not os.path.exists(fig_folder_path):    
        os.makedirs(fig_folder_path)
        
    if dataset_based:
        fig_mse_change_name = f'{data_tag}_mse_change_{points}_Together'
    else:
        fig_mse_change_name = f'mse_change_{points}_Together'
        
    plt.savefig(fig_folder_path + fig_mse_change_name + '.svg')
    plt.savefig(fig_folder_path + fig_mse_change_name + '.pdf')
    plt.savefig(fig_folder_path + fig_mse_change_name + '.png')
    plt.close()
    
if plot_mode == 'Separate':
    plot_separate()
elif plot_mode == 'Together':
    plot_together()
    plot_mse()
    # plot_mse_change()

# plot_separate()
