
import numpy as np
import pandas as pd

from utils.timefeatures import time_features

class DataStamp():
    def __init__(self, df, args):
        self.df = df
        self.args = args
        self.timeenc = 0 if self.args.embed != 'timeF' else 1
        self.freq = self.args.freq
        
    def create_data_stamps(self):
        # Create the stamp dataframe
        df_stamp = self.df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
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


        self.data_stamp = data_stamp
        # with open('data_stamp.pkl', 'wb') as file:
            # pickle.dump(self.data_stamp, file)
            
        return self.data_stamp
        
