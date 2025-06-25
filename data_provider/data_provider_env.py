import os
import pandas as pd

import warnings

from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
from sklearn.preprocessing import MinMaxScaler

import joblib

#%% 

class Dataset_Pred(Dataset):
    def __init__(self, df_predict, root_path, args, scaler=MinMaxScaler(), 
                 is_simulator=False, flag='pred', size=None, features='S', 
                 data_path='ETTh1.csv', target='OT', scale=True, inverse=False,
                 timeenc=0, freq='15min', cols=None, df_raw=None, df_stamp=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.args = args
        self.df_predict = df_predict
        self.scaler = scaler
        self.is_simulator = is_simulator
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.df_raw = df_raw
        self.df_stamp = df_stamp

        self.__read_data__()

    def __read_data__(self):
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(self.df_raw.columns)
            cols.remove(self.target)
            if 'date' in cols:
                cols.remove('date')

        self.df_predict = self.df_predict[['date'] + cols + [self.target]]
        border1 = len(self.df_predict) - self.seq_len
        border2 = len(self.df_predict)

        if self.features in ['M', 'MS']:
            cols_data = self.df_predict.columns[1:]
            df_data = self.df_predict[cols_data]
        elif self.features == 'S':
            df_data = self.df_predict[[self.target]]

        if self.scale:
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = self.df_predict[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        # embed != timeF -> timeenc = 0
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            if self.freq == 't':
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
            
        # embed = timeF -> timeenc = 1
        elif self.timeenc == 1:
            '''
            Will add these to the dataset:
            All of them are encoded as value between [-0.5, 0.5]
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
            MonthOfYear
            '''
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            
        '''
        df_temp = pd.DataFrame(columns=['date'])
        df_temp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        
        index_range = range(self.df_raw.index.get_loc(df_temp.date.iloc[0]),
                            self.df_raw.index.get_loc(df_temp.date.iloc[-1])+1)
        '''
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

#%%

def data_provider(args, flag, df_predict, scaler, df_raw, df_stamp):
    shuffle_flag = False
    drop_last = False
    batch_size = 1
    freq = args.freq
    Data = Dataset_Pred
    timeenc = 0 if args.embed != 'timeF' else 1
    scale = args.scale
    is_simulator = args.simulator
    num_workers = 2 # args.num_workers

    data_set = Data(
        df_predict=df_predict,
        root_path=args.root_path,
        scaler=scaler,
        is_simulator=is_simulator,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        scale=scale,
        args=args,
        df_raw=df_raw,
        df_stamp=df_stamp
        )
        
    # print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    
    return data_set, data_loader
