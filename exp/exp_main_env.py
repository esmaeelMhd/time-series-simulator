import os
import torch

import warnings

import numpy as np
import pandas as pd
from data_provider.data_provider_env import data_provider

import joblib

#%%

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            # print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

#%%

class Exp_Main(Exp_Basic):
    def __init__(self, args, model, df_raw):
        super(Exp_Main, self).__init__(args)
        self.model = model
        # Load the scaler
        SCALER_PATH = './scalers/' + self.args.setting + '/'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.scaler = joblib.load(SCALER_PATH + 'scaler.gz')
        
        self.df_raw = df_raw
        self.df_stamp = None
        
    def _get_data(self, flag):
        data_set, data_loader = data_provider(args=self.args, 
                                              flag='pred', 
                                              df_predict=self.df_predict, 
                                              scaler=self.scaler,
                                              df_raw=self.df_raw,
                                              df_stamp=self.df_stamp)
        return data_set, data_loader
    
    def predict(self, df_predict, setting, load=False):
        self.df_predict = df_predict
        pred_data, pred_loader = self._get_data(flag='pred')
    
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
    
        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
    
        preds = np.array(preds)       
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        preds = preds.reshape(preds.shape[1], preds.shape[2])
        '''
        SCALER_PATH = './scalers/' + self.args.setting + '/'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.scaler = joblib.load(SCALER_PATH + 'scaler.gz')
        preds_inv = self.scaler.inverse_transform(preds)
        
        # Reshape it from [1, pred_len, in_features] to [pred_len, in_features]
        preds_inv = np.array(preds_inv).reshape(np.array(preds_inv).shape[0], -1)
        '''
        return preds #preds_inv

