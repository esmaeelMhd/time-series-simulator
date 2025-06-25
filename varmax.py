import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt as sqrt
import pytz
import datetime
from sklearn.model_selection import train_test_split
import os

from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.varmax import VARMAXResults

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, AutoLocator)
plt.clf()

from utils.env_helper import EnvHelper

from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation
from sysidentpy.utils.save_load import save_model, load_model


plt.switch_backend('agg')

#%%

dataset_name = 'wastewater.csv'
df_raw = pd.read_csv('./datasets/' + dataset_name)

df_raw['date'] = pd.to_datetime(df_raw.date)
df_raw = df_raw.set_index(['date'])

data = df_raw.copy(deep=True)

control_var = 'IN_METAL_Q'
objective_var = 'T1_PO4'

data = data[:int(0.01*len(data))]

#creating the train and validation set
df_train, df_test = train_test_split(data, test_size=0.1, shuffle=False)

#%% 

def VARMAX_model(train, chkpt_name):
    print('Training the model ...')
    # fit model
    p = seq_len - 1
    q = 1
    model = VARMAX(train.drop(control_var, axis=1), exog=train[control_var], order=(p, q), freq='min')
    model_fit = model.fit(disp=False)
    model_fit.save('./checkpoints/' + chkpt_name+'.pkl')

def NARMAX_model(train, chkpt_name):
    print('Training the model ...')
    x_train = np.array(train[control_var]).reshape(len(train), 1)
    y_train = np.array(train[objective_var]).reshape(len(train), 1)
        
    basis_function = Polynomial(degree=2)
    model = FROLS(
        order_selection=True,
        n_info_values=10,
        extended_least_squares=False,
        ylag=seq_len,
        xlag=seq_len,
        info_criteria='aic',
        estimator='least_squares',
        basis_function=basis_function
        )
    model.fit(X=x_train, y=y_train)
    save_model(model=model, path='./checkpoints/', file_name=chkpt_name+'.syspy')

#%% 
   
def test_model(model, test):
    print('Testing the model ...')
    if model == 'varmax':
        model_fit = VARMAXResults.load('./checkpoints/' + model + '.pkl')
        # make prediction
        yhat = model_fit.forecast(steps=len(test), exog=test[control_var])
        res = pd.DataFrame()
        for col in test:
            if col == control_var:
                res[col] = test[col].values
            else:
                res[col] = test[col].values
                res['pred_' + col] = yhat[col].values
                
    elif model == 'narmax':
        model = load_model(path='./checkpoints/', file_name='narmax.syspy')
        x_test = np.array(test.iloc[:, test.columns.get_loc(control_var)]).reshape(len(test), 1)
        y_test = np.array(test.iloc[:, test.columns.get_loc(objective_var)]).reshape(len(test), 1)
        yhat = model.predict(X=x_test, y=y_test)
        mse = mean_squared_error(y_test, yhat)
        print('mse: ', mse)
        
        res = pd.DataFrame()
        for col in test:
            if col == control_var:
                res[col] = test.iloc[seq_len:, test.columns.get_loc(col)].values
            elif col == objective_var:
                res[col] = test.iloc[seq_len:, test.columns.get_loc(col)].values
                res['pred_' + col] = yhat[seq_len:]
        
        ee = compute_residues_autocorrelation(y_test, yhat)
        #plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
        x1e = compute_cross_correlation(y_test, yhat, x_test)
        #plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")

        
    return res

def show_graph(results_dict, fig_name):
    if episode_length > 360:
        columns = 1
        rows = len(results_dict.keys())
        wspace = 0.3
        hspace = 0.5
        figsize = (6, 1.5*rows)
    else:
        if len(results_dict.keys()) < 3:
            columns = len(results_dict.keys())
            rows = 1
            figsize = (7, 4*rows) 
        else:
            columns = 2
            rows = len(results_dict.keys())//2 if len(results_dict.keys())%2 == 0 else len(results_dict.keys())//2 + 1
            figsize = (6.5, 2.2*rows) 

        wspace = 0.5
        hspace = 0.3
    fig, axs = plt.subplots(rows, columns, figsize=figsize, dpi=500)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    for (point, ax) in zip(results_dict.keys(), axs.ravel()):
        freq = datetime.timedelta(minutes=1)
        dates = pd.date_range(start=pd.to_datetime(point), end=pd.to_datetime(point) + episode_length*freq, freq='min')
        dates = pd.to_datetime(dates)
        times = dates.strftime('%H:%M')
        data = results_dict[point]
        data = data.drop([col for col in data.columns if 'T1_PO4' not in col], axis=1)
        # data.reset_index(inplace=True, drop=True)
        ax.set_title(point)
        if i == ((rows-1)*columns):
            ax.set_xlabel('Time')
            ax.set_ylabel('P-amount [mg/L]')
        
        for col in data.columns:
            if col.lower().startswith('pred'):
                ax.plot(data[col], label=col, linestyle="dotted")
            else:
                ax.plot(data[col], label=col)
        
        ax.set_xticks(np.arange(len(times)))
        ax.set_xticklabels(times)
        ax.xaxis.set_major_locator(MultipleLocator(episode_length/6))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.legend()

    plt.savefig('./figures/' + fig_name + '.png')
    plt.savefig('./figures/' + fig_name + '.pdf')
    plt.close('all')
    plt.clf()


#%%

model = 'narmax'
helper = EnvHelper()
tf = 'Seasons'
day_of_m = 'Middle'
t_of_day = 'Morning'
f_date = pd.to_datetime('2021-09')
l_date = pd.to_datetime('2022-08')
l_points, p_names = list_points, points_names = helper.make_points(test_frequency=tf,
                                                                   day_of_the_month=day_of_m,
                                                                   time_of_the_day=t_of_day,
                                                                   first_date=f_date,
                                                                   last_date=l_date)
seq_len = 240
episode_length = 360
chkpt_name = f'model_{seq_len}Seq'

if model == 'narmax':
    if not os.path.exists('./checkpoints/' + chkpt_name + '.syspy'):
        NARMAX_model(df_train, chkpt_name)
elif model == 'varmax':
    if not os.path.exists('./checkpoints/' + chkpt_name + '.pkl'):
        VARMAX_model(df_train, chkpt_name)
        
for i, point in enumerate(l_points):
    p_name = p_names[i]
    start_idx = df_raw.index.get_loc(point) - seq_len
    end_idx = start_idx + episode_length + seq_len
    df_test = df_raw.iloc[start_idx:end_idx]
    
    if model == 'narmax':
        narmax_dict = {key:None for key in p_names}
        df_narmax = test_model('narmax', df_test)
        narmax_dict[p_name] = df_narmax
        show_graph(narmax_dict, 'narmax_predictions')
        
    elif model == 'varmax':
        varmax_dict = {key:None for key in p_names}
        df_varmax = test_model('varmax', df_test)
        varmax_dict[p_name] = df_varmax
        show_graph(varmax_dict, 'varmax_predictions')

        
