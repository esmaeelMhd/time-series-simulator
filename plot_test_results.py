
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
import pickle
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import os
import warnings
from matplotlib.ticker import NullLocator


#%% Plot and device options

#plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.color"] = "black"
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.xmargin'] = 0.02
plt.rcParams['axes.ymargin'] = 0.04
plt.rcParams['path.simplify'] = True
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)
fig_dpi = 1000
plt.switch_backend('agg')
# plt.switch_backend('qt5agg')

#%%

plot_single = False
df = pd.read_csv('./datasets/wastewater.csv')

plot_start = pd.to_datetime('2022-12-01 00:00:00+00:00')
plot_end = pd.to_datetime('2022-12-06 00:01:00+00:00')

setting = 'Phosphorous_DLinear_custom_ftM_sl240_ll48_pl1_dm128_nh8_el2_dl2_df2048_fc3_ebtimeF_dtTrue_Exp_scaleTrue_0'

result_folders = list([x for x in os.listdir('./results/') if 'New_CorrH' not in x])
list_metrics = list(['mae', 'mse', 'rmse', 'rse', 'corr'])
df_metrics = pd.DataFrame(columns = list_metrics)
df_metrics['names'] = result_folders
df_metrics = df_metrics.set_index(['names'])

for name in result_folders:
    if 'metrics.npy' in os.listdir('./results/' + name):
        metrics = np.load('./results/' + name + '/metrics.npy', allow_pickle=True)
        if len(metrics) == 7:
            metrics = np.delete(metrics, 3)
            metrics = np.delete(metrics, 4)
        #metrics = metrics.astype(float)
        df_metrics.loc[name, :] = metrics

best_models = {'LSTM':'',
               'Transformer':'',
               'Informer':'',
               'Autoformer':'',
               'DLinear':'',
               'NLinear':'',   
               }

for model in best_models.keys():
    items = list([x for x in df_metrics.index if model in x])
    df_metrics['mse'] = pd.to_numeric(df_metrics['mse'])
    list_mse = df_metrics.loc[items, 'mse']
    min_mse_idx = list_mse.idxmin()
    best_models[model] = min_mse_idx
    
#%%

n_features = len(df.columns) - 1

def inverse_transform(scaler, arr, args):
    if args.model == 'LSTM' and args.embed == 'timeF':
        if 'time_scaled' in args and args.time_scaled == 'Unscaled':
            arr = scaler.inverse_transform(arr)
        elif 'time_scaled' in args and args.time_scaled == 'Scaled':
            arr = np.concatenate((arr, np.ones((arr.shape[0], 6))), axis = 1)            
            arr = scaler.inverse_transform(arr)
            arr = arr[:,:n_features]
    else:
        arr = scaler.inverse_transform(arr)
    return arr

def format_predictions(preds, vals, args, df_test):
    index = df_test.index[-len(vals):]
    target_idx = df_test.columns.get_loc(args.target)
    df_result = pd.DataFrame(data={"value": vals[:,target_idx], 
                                   "prediction": preds[:,target_idx]}, 
                                    index=index)
    
    df_result = df_result.sort_index()
    return df_result  
    
#%%

def plot_results(model, df_result, ax):
    
    colors_dict = {'LSTM': 'blue', 'Transformer': 'orange', 'Informer': 'green',
                  'Autoformer': 'red', 'DLinear': 'purple', 'NLinear': 'brown'}
    
    # ax.set_xlabel('Date', labelpad=5, fontsize=8)
    # ax.set_ylabel('P-amount [mg/L]', labelpad=5, fontsize=8)

    axins = ax.inset_axes((0.05, 0.65, 0.2, 0.3))

    month_day_formatter = mdates.DateFormatter('%b %d')
    day_locator = mdates.DayLocator(interval=2)
    ax.xaxis.set_major_locator(day_locator)
    ax.xaxis.set_major_formatter(month_day_formatter)
    ax.tick_params(which='major', length=2)

    
    # use formatters to specify major and minor ticks
    # axins.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    # axins.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    
    plot_color = colors_dict.get(model)
    ax.plot(df_result.prediction, linewidth=0.5, color=plot_color, label='Predicted P')
    ax.plot(df_result.value, '.', markersize=0.5, color='black', label='Actual P')
    ax.grid(visible=True, which='major', color='silver', alpha=0.25, linewidth=0.075)
    ax.grid(visible=True, which='minor', color='silver', alpha=0.25, linewidth=0.075)
    
    axins.plot(df_result.prediction, linewidth=0.5, color=plot_color, label='Predicted P')
    axins.plot(df_result.value, '.', markersize=0.2, color='black', label='Actual P')
    
    ax.legend(fontsize=8)

    inset_start = pd.to_datetime('2022-12-03 00:00:00+00:00')
    inset_end = pd.to_datetime('2022-12-03 04:00:00+00:00')
    
    axins.xaxis.set_major_locator(NullLocator())
    axins.yaxis.set_major_locator(NullLocator())

    axins.set_xlim(inset_start, inset_end)
    axins.set_ylim(0, 4.5)
    ax.indicate_inset_zoom(axins)

#%% 

def get_df_result(name):
    ARGS_PATH = './args/' + name + '/'
    # Load args
    with open(ARGS_PATH + 'args.pkl', 'rb') as file:
        args = pickle.load(file)
    
    RESULTS_PATH = './results/' + name + '/'
    preds = np.load(RESULTS_PATH + 'pred.npy')
    true = np.load(RESULTS_PATH + 'true.npy')
    
    n_features = len(df.columns) - 1

    if 'out_features' in args:
        n_features = args.out_features
    
    preds = preds.reshape(preds.shape[0], n_features)
    true = true.reshape(true.shape[0], n_features)
    
    SCALER_PATH = './scalers/' + name + '/'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        scaler = joblib.load(SCALER_PATH + 'scaler.gz')
    
    df_test = pd.read_pickle(RESULTS_PATH + 'df_test.pkl')
    df_dates.loc[name, 'start_date'] = df_test.index[0]
    
    preds = inverse_transform(scaler, preds, args)
    true = inverse_transform(scaler, true, args)
    
    df_result = format_predictions(preds, true, args, df_test)
    df_result = df_result[plot_start:plot_end]
    
    return df_result, args

#%%

df_dates = pd.DataFrame(columns=['names','start_date'])
df_dates['names'] = result_folders
df_dates = df_dates.set_index(['names'])

if plot_single:
    names_list = list([name for name in result_folders]) 
    for name in names_list:
        df_result, args = get_df_result(name)
        mse = df_metrics.loc[name, 'mse']
    
        # Define figure
        fig = plt.figure(figsize=(12, 6), dpi=fig_dpi)
        fig.suptitle(name + ' | ' + f'MSE: {mse}', fontweight="bold")
        ax = fig.gca()
        plot_results(name, df_result, ax)
        
        plt.ion()
        plt.show()
        
        FIG_PATH = './results/' + name + '/'
        if not os.path.exists(FIG_PATH):    
            os.makedirs(FIG_PATH)
        
        fig_name = 'test_results'
        plt.savefig(FIG_PATH + fig_name + '_' + str(fig.get_dpi()) + 'dpi' + '.svg')
        plt.savefig(FIG_PATH + fig_name + '_' + str(fig.get_dpi()) + 'dpi' +'.pdf') 
        plt.savefig(FIG_PATH + fig_name + '_' + str(fig.get_dpi()) + 'dpi' + '.png')
    
else:
    names_list = list([name for name in best_models.values()]) 
    # Define figure
    fig, axs = plt.subplots(3,2, figsize=(7, 3.6), dpi=fig_dpi)
    # fig.tight_layout(pad=4.0)
    list_dataframes = []
    for name, ax in zip(names_list, axs.ravel()):
        df_result, args = get_df_result(name)
        list_dataframes.append(df_result)
        mse = df_metrics.loc[name, 'mse']
        
        plot_results(args.model, df_result, ax)
        model = list(best_models.keys())[list(best_models.values()).index(name)]
        print(model)
        plot_title = f'{model} [MSE = {mse:0.4f}]'
        ax.title.set_text(plot_title)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.55)
    #fig.subplots_adjust(top=0.92, bottom=0.1, right=0.95, left=0.1)

    for i in range(axs.shape[1]):   
        axs[-1, i].set_xlabel('Date', labelpad=5, fontsize=8)
        
    for j in range(axs.shape[0]):
        axs[j, 0].set_ylabel('P-amount [mg/L]', labelpad=5, fontsize=8)

    FIG_PATH = './figures/' 
    if not os.path.exists(FIG_PATH):    
        os.makedirs(FIG_PATH)
    
    fig_name = 'test_results_together'
    plt.savefig(FIG_PATH + fig_name + '_' + str(fig.get_dpi()) + 'dpi' + '.svg')
    plt.savefig(FIG_PATH + fig_name + '_' + str(fig.get_dpi()) + 'dpi' +'.pdf') 
    plt.savefig(FIG_PATH + fig_name + '_' + str(fig.get_dpi()) + 'dpi' + '.png')
    
    plt.ion()
    plt.show()

    
#%%
# Collect all the font names available to matplotlib
# font_names = [f.name for f in fm.fontManager.ttflist]
# print(font_names)