"""
Created on Monday July 18 2022
@author: Esmaeel Mohammadi

# =============================================================================
# This class is used to train, test, and validate the LSTM model:
    1. Training the model
    2. Validating the train results
    3. Prediction using the test dataset
    4. Prediction of the future
    5. Retraining the model using its own prediction
    6. Validation of the retrain results
    7. Test the simulation environment for the retrained model
# =============================================================================
"""

import numpy as np
import torch
import logging
import datetime
import csv
import os
from tqdm import tqdm

from utils.tools import EarlyStopping, EarlyStoppingRetrain, adjust_learning_rate
from utils.metrics import metric
from utils.dilate_loss import dilate_loss
from torch.utils.data import TensorDataset, DataLoader
from tslearn.metrics import dtw, dtw_path
# from apex import amp
import time, gc
start_time = None

#%% Optimization Class

class Optimization:
    """
    Initialization of the Optimization class for LSTM model operations.

    Parameters:
    - model (LSTMModel): LSTM model.
    - loss_fn (torch.nn.modules.Loss): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer.
    - args (Namespace): Training arguments.
    - setting (str): Model setting name.
    - device: Training device (CPU/GPU).
    - is_retrain (bool, optional): Indicates if the model is in retraining mode. Default is False.

    Attributes:
    - train_losses (list[float]): Loss values from training.
    - val_losses (list[float]): Loss values from validation.
    - [Other attributes as per the original structure]
    """
    def __init__(self, initiator=None, model=None, loss_fn=None, optimizer=None, args=None,
                 setting=None, device=None, target_idx=-1, is_retrain=False, is_policy=False):
        self.initiator = initiator
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.args = args
        self.setting = setting
        self.device = device
        self.target_idx = target_idx
        self.is_retrain = is_retrain
        self.is_policy = is_policy
        self.train_losses = []
        self.train_losses_shape = []
        self.train_losses_temporal = []
        self.val_losses = []
        self.val_losses_dtw = []
        self.val_losses_tdi = []
        self.train_ys = []
        
        # Setup the device based on GPU usage
        self.device = torch.device('cuda') if torch.cuda.is_available() and self.args.use_gpu else torch.device('cpu')
        if self.device == 'cuda':
            self.device_ids = ','.join(str(i) for i in range(torch.cuda.device_count()))
            self.use_multi_gpu = torch.cuda.device_count() > 1
        else:
            self.device_ids = ''
            self.use_multi_gpu = False  
        
        self.target_idx = 0 if self.is_policy else -1
                        
        self.ctrl_vars = []
        if hasattr(self.args, 'ctrl_vars'):
            if isinstance(self.args.ctrl_vars, str):
                self.ctrl_vars.append(self.args.ctrl_vars)
            elif isinstance(self.args.ctrl_vars, list):
                self.ctrl_vars = self.args.ctrl_vars
        elif hasattr(self.args, 'control_variable'):
            if isinstance(self.args.control_variable, str):
                self.ctrl_vars.append(self.args.control_variable)
            elif isinstance(self.args.control_variable, list):
                self.ctrl_vars = self.args.control_variable
        
        self.ind_vars = []
        if hasattr(self.args, 'ind_vars'):
            if isinstance(self.args.ind_vars, str):
                self.ind_vars.append(self.args.ind_vars)
            elif isinstance(self.args.ind_vars, list):
                self.ind_vars = self.args.ind_vars
        elif hasattr(self.args, 'independent_vars'):
            if isinstance(self.args.independent_vars, str):
                self.ind_vars.append(self.args.independent_vars)
            elif isinstance(self.args.independent_vars, list):
                self.ind_vars = self.args.independent_vars
            
        self.num_time_f = 6
        if hasattr(self.args, 'num_time_f'):
            self.num_time_f = self.args.num_time_f
            
        self.n_actions = len(self.ctrl_vars)
        self.n_ind = len(self.ind_vars)
        
        if self.is_policy:
            self.out_features = self.n_actions
        else:
            self.out_features = self.args.out_features
     
    def start_timer(self):
        """
        Starts a timer for performance measurement.
    
        It performs garbage collection and resets CUDA memory stats.
        """
        global start_time
        gc.collect()  # Garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache
            torch.cuda.reset_max_memory_allocated()  # Reset memory usage stats
            torch.cuda.synchronize()  # Synchronize CUDA operations
        start_time = time.time()

    
    def end_timer_and_print(self, local_msg):
        """
        Ends the timer and prints the elapsed time.
    
        Parameters:
        - local_msg (str): Custom message to print along with the timer results.
        """
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        total_time = end_time - start_time
        epochs = self.retrain_args.retrain_epochs if self.is_retrain else self.args.train_epochs
        print(f"\n{local_msg}")
        print(f"Total execution time = {int(total_time // 60)} min {int(total_time % 60)} sec")
        print(f"Execution time per epoch = {int((total_time // epochs) // 60)} min {int((total_time // epochs) % 60)} sec")
        if torch.cuda.is_available():
            print(f"Max memory used by tensors = {torch.cuda.max_memory_allocated()} bytes")


    def setup_logging(self, log_name):
        """
        Sets up logging with the specified log file name.
    
        Parameters:
        - log_name (str): Name of the log file.
        """
        logging.basicConfig(filename=log_name, level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    def create_metrics_dict(self, training_loss, validation_loss):
        """
        Creates a dictionary of metrics for logging and writing to CSV.
    
        Parameters:
        - training_loss: Training loss value.
        - validation_loss: Validation loss value.
    
        Returns:
        - dict: Dictionary containing metrics data.
        """
        metrics_dict = {'Name': self.args.setting, 
                        'Date': str(datetime.datetime.now()),
                        'Model': self.args.model,
                        'Data': self.args.data_tag,
                        'Features': self.args.in_features,
                        'Time Scale': self.args.time_scaled,
                        'Seq Len': self.args.seq_len,
                        'Pred Len': self.args.pred_len,
                        'Batch Size': self.args.batch_size,
                        'LR': self.args.learning_rate,
                        'Hidden Dim': self.args.hidden_dim,
                        'Layers': self.args.layer_dim,
                        'Train Loss': training_loss,
                        'Validation loss': validation_loss}
        
        return metrics_dict
    
    def write_metrics(self, training_loss, validation_loss):
        """
        Writes training and validation metrics to log and CSV files.
    
        Parameters:
        - training_loss: Training loss value.
        - validation_loss: Validation loss value.
        """
        # log_name = 'retrain_metrics.log' if self.is_retrain else 'train_metrics.log'
        # self.setup_logging(log_name)
    
        metrics_dict = self.create_metrics_dict(training_loss, validation_loss)
        logging.info(metrics_dict)
    
        csv_name = 'retrain_metrics.csv' if self.is_retrain else 'train_metrics.csv'
        fieldnames = metrics_dict.keys()
    
        with open(csv_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:  # Writes the header if the file is empty
                writer.writeheader()
            writer.writerow(metrics_dict)

            
    def compute_dilate_loss(self, y, yhat, dilate_all):
        """
        Computes DILATE loss.
    
        Parameters:
        - y: Targets.
        - yhat: Predictions.
        - dilate_all: If dilate should be calculated for all of the out features or not.
        """
        # Add one dimension if they are 2D
        if y.ndim == 2: 
            y = y.unsqueeze(0) 
        if yhat.ndim == 2:
            yhat = yhat.unsqueeze(0)

        if self.is_retrain:
            # y shape: (retrain length, pred len, out features)
            retrain_length = y.shape[0]
            batch_size = 1
            if dilate_all:
                y = y[:, 0, -self.out_features:].view([1, retrain_length, self.out_features])
                yhat = yhat[:, 0, -self.out_features:].view([1, retrain_length, self.out_features])
            else:
                y = y[:, 0, self.target_idx].view([1, retrain_length, 1])
                yhat = yhat[:, 0, self.target_idx].view([1, retrain_length, 1])
        else:
            # y shape: (batch size, pred len, out features)
            batch_size = self.args.batch_size
            if dilate_all:
                y = y[:, :, -self.out_features:].view([y.shape[0], self.args.pred_len, self.out_features])
                yhat = yhat[:, :, -self.out_features:].view([yhat.shape[0], self.args.pred_len, self.out_features])
            else:
                y = y[:, :, self.target_idx].view([y.shape[0], self.args.pred_len, 1])
                yhat = yhat[:, :, self.target_idx].view([yhat.shape[0], self.args.pred_len, 1])
                
        #self.train_ys.append(np.concatenate((y.detach().cpu().numpy(), yhat.detach().cpu().numpy()), axis=2))

        alpha = self.retrain_args.alpha_dilate if self.is_retrain else self.args.alpha_dilate
        gamma = self.retrain_args.gamma_dilate if self.is_retrain else self.args.gamma_dilate
        '''
        loss, loss_shape, loss_temporal = dilate_loss(y[:, :, self.target_idx].view([batch_size, y.shape[1], 1]),
                                                            yhat[:, :, self.target_idx].view([batch_size, y.shape[1], 1]), 
                                                            alpha, gamma, self.device)  
        
        '''
        d_losses = []
        d_losses_shape = []
        d_losses_temporal = []
        for f in range(y.shape[2]):
            f_loss, f_loss_shape, f_loss_temporal = dilate_loss(y[:, :, f].view([batch_size, y.shape[1], 1]),
                                                                yhat[:, :, f].view([batch_size, y.shape[1], 1]), 
                                                                alpha, gamma, self.device)   
            d_losses.append(f_loss)
            d_losses_shape.append(f_loss_shape)
            d_losses_temporal.append(f_loss_temporal)
        
        loss = torch.stack(d_losses).mean()
        loss_shape = torch.stack(d_losses_shape).mean()
        loss_temporal = torch.stack(d_losses_temporal).mean()
        
        return loss, loss_shape, loss_temporal
    
    def train_step(self, x, y):
        """The method train_step completes one step of training.
    
        Given the features (x) and the target values (y) tensors, the method completes
        one step of the training. First, it activates the train mode to enable back prop.
        After generating predicted values (yhat) by doing forward propagation, it calculates
        the losses by using the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.
    
        Args:
            x (torch.Tensor): Tensor for features to train one step
            y (torch.Tensor): Tensor for target values to calculate losses
        """
        # Sets the type of the loss function (mse or dilate)
        if self.is_retrain:
            loss_type = self.retrain_args.loss_function
            dilate_all = self.retrain_args.dilate_all
        else:
            loss_type = self.args.loss
            dilate_all = self.args.dilate_all
                        
        # Sets model to train mode
        self.model.train()
        
        if self.is_retrain:
            use_amp = self.retrain_args.use_amp
        else:
            use_amp = self.args.use_amp
    
        # Makes predictions
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            yhat = self.model(x)
        
            # Computes loss
            if loss_type == 'mse':
                loss = self.loss_fn(y, yhat)  
                # y_temp = y[:, :, self.target_idx].view([self.args.batch_size, self.args.pred_len, 1]).detach().cpu().numpy()
                # yhat_temp = yhat[:, :, self.target_idx].view([self.args.batch_size, self.args.pred_len, 1]).detach().cpu().numpy()
                # self.train_ys.append(np.concatenate((y_temp, yhat_temp), axis=2))                 
            
            if loss_type == 'dilate':
                loss, loss_shape, loss_temporal = self.compute_dilate_loss(y, yhat, dilate_all)
                self.train_ys.append(np.concatenate((y.detach().cpu(), yhat.detach().cpu()), axis=2))                 
    
        self.amp_scaler.scale(loss).backward()
        self.amp_scaler.step(self.optimizer)
        self.amp_scaler.update()
        self.optimizer.zero_grad() # set_to_none=True here can modestly improve performance
        
        # Backpropagation and parameters update
        '''
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        '''
    
        # Detaches the tensors and moves them to the CPU
        y = y.detach().cpu()
        yhat = yhat.detach().cpu()
    
        # Returns the loss
        if loss_type == 'mse':
            return loss.item()
        if loss_type == 'dilate':
            return loss.item(), loss_shape.item(), loss_temporal.item()            


    def train(self, train_loader, val_loader):
        """The method train performs the model training

        The method takes DataLoaders for training and validation datasets, batch size for
        mini-batch training, number of epochs to train, and number of features as inputs.
        Then, it carries out the training by iteratively calling the method train_step for
        train_epochs times. If early stopping is enabled, then it  checks the stopping condition
        to decide whether the training needs to halt before train_epochs steps. Finally, it saves
        the model in a designated file path.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
            val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
        """
        # Creates a folder for saving the best model
        checkpoints = self.args.checkpoints
        chkpt_path = os.path.join(checkpoints, self.setting)
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)
        
        # Setting up the device from input args
        device = self.device
        # Setting the loss function type
        loss_type = self.args.loss
        
        # Making an instance of EarlyStopping class
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, initiator=self.initiator)
        
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self.start_timer()
        
        # Start training
        for epoch in range(1, self.args.train_epochs + 1):
            batch_losses = []
            batch_losses_shape = []
            batch_losses_temporal = []

            for x_batch, y_batch in train_loader:
                # Reshaping x_batch to: [batch_size, sequence_length, n_features]
                x_batch = x_batch.view([self.args.batch_size, -1, self.args.in_features]).to(device)
                y_batch = y_batch.to(device)
                
                loss, loss_shape, loss_temporal = torch.tensor(0, device=device), torch.tensor(0, device=device), torch.tensor(0, device=device)
                if loss_type == 'mse':
                    loss = self.train_step(x_batch, y_batch)
                    batch_losses.append(loss)

                elif loss_type == 'dilate':
                    loss, loss_shape, loss_temporal = self.train_step(x_batch, y_batch)   
                    batch_losses.append(loss)
                    batch_losses_shape.append(loss_shape)
                    batch_losses_temporal.append(loss_temporal)
                
            training_loss = np.mean(batch_losses)
            training_loss_shape = np.mean(batch_losses_shape)
            training_loss_temporal = np.mean(batch_losses_temporal)
            self.record_train_losses(training_loss, training_loss_shape, training_loss_temporal)

            with torch.no_grad():
                batch_val_losses = []
                batch_val_losses_dtw = []
                batch_val_losses_tdi = []
                for x_val, y_val in val_loader:
                    # Reshaping x_val to: [batch_size, sequence_length, n_features]
                    x_val = x_val.view([self.args.batch_size, -1, self.args.in_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val.detach().cpu(), yhat.detach().cpu()).item()
                    # DTW and TDI
                    val_loss_dtw, val_loss_tdi = 0, 0
                    for k in range(self.args.batch_size):   
                        batch_size, N_output = y_val[:, :, self.target_idx].view([self.args.batch_size, self.args.pred_len, 1]).shape[0:2]
                        target = y_val[:, :, self.target_idx].view([self.args.batch_size, self.args.pred_len, 1])
                        outputs = yhat[:, :, self.target_idx].view([self.args.batch_size, self.args.pred_len, 1])
                        target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
                        output_k_cpu = outputs[k,:,0:1].view(-1).detach().cpu().numpy()

                        path, sim = dtw_path(target_k_cpu, output_k_cpu)   
                        val_loss_dtw += sim
                                   
                        Dist = 0
                        for i,j in path:
                                Dist += (i-j)*(i-j)
                        val_loss_tdi += Dist / (N_output*N_output)            
                                    
                    val_loss_dtw = val_loss_dtw / batch_size
                    val_loss_tdi = val_loss_tdi / batch_size
                    batch_val_losses.append(val_loss)
                    batch_val_losses_dtw.append(val_loss_dtw)
                    batch_val_losses_tdi.append(val_loss_tdi)
                validation_loss = np.mean(batch_val_losses)
                validation_loss_dtw = np.mean(batch_val_losses_dtw)
                validation_loss_tdi = np.mean(batch_val_losses_tdi)
                self.val_losses.append(validation_loss)
                self.val_losses_dtw.append(validation_loss_dtw)
                self.val_losses_tdi.append(validation_loss_tdi)

            if (epoch <= 10) | (epoch % 50 == 0):
                if loss_type == 'mse':
                    print(
                        f"[{epoch}/{self.args.train_epochs}] Training loss: {training_loss:.4f} | Validation loss: {validation_loss:.4f}"
                        )
                if loss_type == 'dilate':
                    print(
                        f"[{epoch}/{self.args.train_epochs}] Training loss: {training_loss:.4f} | shape: {training_loss_shape:.4f} | " +\
                        f"temporal: {training_loss_temporal:.4f} | Validation loss: {validation_loss:.4f} | dtw: {validation_loss_dtw:.4f}\t" +\
                        f"tdi: {validation_loss_tdi:.4f}"
                        )
            
            # Early Stopping
            if self.args.do_early_stop:
                early_stopping(validation_loss, self.model, chkpt_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    self.write_metrics(training_loss, validation_loss)
                    break
            else:
                torch.save(self.model.state_dict(), chkpt_path + '/' + 'checkpoint.pth')
            
            # Learning rate Optimization
            if self.args.do_lr_opt == True and epoch < self.args.train_epochs:
                adjust_learning_rate(self.optimizer, epoch + 1, self.args)
            
            gc.collect()
                
        self.end_timer_and_print(f"Mixed precision: {self.args.use_amp}")
        self.write_metrics(training_loss, validation_loss)
        gc.collect()

    def record_train_losses(self, training_loss, training_loss_shape, training_loss_temporal):
        self.train_losses.append(training_loss)
        self.train_losses_shape.append(training_loss_shape)
        self.train_losses_temporal.append(training_loss_temporal)
        
    def evaluate(self, test_loader, setting, batch_size=1, n_features=1):
        """The method evaluate performs the model evaluation

        The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
        and number of features as inputs. Similar to the model validation, it iteratively
        predicts the target values and calculates losses. Then, it returns two lists that
        hold the predictions and the actual values.

        Note:
            This method assumes that the prediction from the previous step is available at
            the time of the prediction, and only does one-step prediction into the future.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            setting (string): Name of the model's folder
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns

        Returns:
            list[float]: The values predicted by the model
            list[float]: The actual values in the test set

        """
        # Setting the device from input args
        device = self.device
        
        # Loading the model
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth'), map_location=self.device))

        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                # Reshaping x_test to: [batch_size, sequence_length, n_features]
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.detach().cpu().numpy())
                values.append(y_test.detach().cpu().numpy())

        # Saving the results
        if self.is_policy:
            folder_path = './policy_results/' + setting + '/'
        else:
            folder_path = './results/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        mae, mse, rmse, rse, corr = metric(np.array(predictions), np.array(values))
        print('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f = open("train_results.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, rse, corr], dtype=object))
        np.save(folder_path + 'pred.npy', predictions)
        np.save(folder_path + 'true.npy', values)
        
        return predictions, values
    
    # Predicting Future Values    
    def forecast_with_predictors(self, forecast_loader, batch_size=1, n_features=1, n_steps=100):
        """Forecasts values for LSTMs with predictors

        The method takes DataLoader for the test dataset, batch size for mini-batch testing,
        number of features and number of steps to predict as inputs. Then it generates the
        future values for LSTM with output for the given n_steps. It uses the
        values from the predictors columns (features) to forecast the future values.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns
            n_steps (int): Number of steps to predict future values

        Returns:
            list[float]: The values predicted by the model
        """
        # Setting the device from input args
        device = self.device

        step = 0
        with torch.no_grad():
            predictions = []
            for x_test, _ in forecast_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.cpu().detach().numpy())

                step += 1
                if step == n_steps:
                    break

        return predictions
    
    # Retraining the model
    def retrain(self, retrain_args, model, retrain_name, retrain_data, retrain_targets, 
                data_idxs, val_loader, test_loader, writer=None):
        """The method retrain performs the model retraining

        The method takes retrain args, the trained model, the name and DataLoaders for 
        retraining and validation datasets. Then, it carries out the retraining by making input datasets 
        including the model's own prediction of previous steps and then iteratively calling the method 
        train_step for retrain_epochs times. If early stopping is enabled, then it  checks the stopping 
        condition to decide whether the retraining needs to halt before retrain_epochs steps. Finally, 
        it saves the model in a designated file path.

        Args:
            retrain_args (Namespace): Args for the model retraining
            model : The loaded model from the checkpoint  
            retrain_name (string): Name of the retrain checkpoint
            retrain_dataset (list): Retrain inputs and targets
            data_idxs (list): Indices of the input and targets from the list
            val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
            test_loader (torch.utils.data.DataLoader): DataLoader that stores simulation test data
        """
        # Start retraining
        self.is_retrain = True
        self.retrain_args = retrain_args
        self.model = model
        experiment = retrain_args.experiment
        epochs = retrain_args.retrain_epochs
        patience = retrain_args.patience
        retrain_mode = retrain_args.retrain_mode
        use_actual_actions = retrain_args.use_actual_actions
        use_actual_ind = retrain_args.use_actual_ind
        do_early_stop = retrain_args.do_early_stop
        loss_type = retrain_args.loss_function
        val_loss_type = retrain_args.val_loss_type
        best_loss = None
        checkpoint_path = self.args.checkpoints + self.args.setting + '/'
        
        '''
        self.retrain_dict = {'x':[],
                            'targets':[],
                            'val_x':[],
                            'y_val':[],
                            'y_pred':[],
                            'val_losses':[],
                            'targets_3d':[]}
        '''
        
        print(f'Number of batches for each epoch: {len(retrain_data)}')
        
        # Making an instance of EarlyStopping class
        early_stopping = EarlyStoppingRetrain(patience=patience, verbose=True)
        
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.retrain_args.use_amp)
        self.start_timer()
        
        # Start retraining
        for epoch in range(1, epochs+1):
            batch_losses = []
            batch_losses_shape = []
            batch_losses_temporal = []

            for i, (data_item, target_item) in enumerate(tqdm(zip(retrain_data, retrain_targets))):
                # batch_idx = data_idxs[i] if experiment == 3 or experiment == 4 else i
                batch_data = torch.Tensor(np.array(data_item)).unsqueeze(0).to(self.device)
                batch_targets = torch.Tensor(np.array(target_item)).unsqueeze(0).to(self.device)
                
                retrain_dataset = TensorDataset(batch_data, batch_targets)
                retrain_loader = DataLoader(retrain_dataset, batch_size=1, 
                                          shuffle=False, drop_last=False)
                   
                for x, targets in retrain_loader:
                    x = x.to(self.device)
                    targets = targets.to(self.device)
                    batch_size = targets.shape[1] -self.args.pred_len + 1
                    targets_3d = torch.empty(batch_size, self.args.pred_len, self.args.in_features)
                    for i in range(batch_size):
                        targets_3d[i, :, :] = targets[:, i:i+self.args.pred_len, :]
                    targets = targets_3d
                    #self.retrain_dict['targets_3d'].append(np.array(targets_3d.detach().cpu()))
                    pred = None
                    losses = []
                    losses_shape = []
                    losses_temporal = []
                    sum_losses = 0
                    
                    if retrain_mode == 'single_pred':
                        for i in range(batch_size):
                            y_true = targets[i, :, :].unsqueeze(0).to(self.device)
                            if pred is not None:
                                pred = torch.cat((pred, y_true[:, :, self.args.out_features:]), dim=2)
                                x = torch.cat([x, pred], dim=1)[:, 1:, :].to(self.device)
                            if loss_type == 'mse':
                                loss = self.train_step(x, y_true[:, :, :self.args.out_features])
                            if loss_type == 'dilate':
                                loss, loss_shape, loss_temporal = self.train_step(x, y_true[:, :, :self.args.out_features])
                                losses_shape.append(loss_shape)
                                losses_temporal.append(loss_temporal)
                            losses.append(loss)    
                            sum_losses += loss
                            
                            with torch.no_grad():
                                self.model.eval()
                                pred = self.model(x)[:, 0, :].view(1, 1, self.args.out_features)
                                if use_actual_actions:
                                    if self.args.out_features < self.args.in_features - self.num_time_f:
                                        if use_actual_ind:
                                            pred = torch.cat((y_true[:, :, :self.n_actions+self.n_ind], pred), dim=2)
                                        else:
                                            pred = torch.cat((y_true[:, :, :self.n_actions], pred), dim=2)
                                    else:
                                        if use_actual_ind:
                                            pred[:, :, :self.n_actions+self.n_ind] = y_true[:, :, :self.n_actions+self.n_ind] 
                                        else:
                                            pred[:, :, :self.n_actions] = y_true[:, :, :self.n_actions] 
                                            
                        # Losses of the whole episode
                        batch_losses.append(np.mean(losses))
                        batch_losses_shape.append(np.mean(losses_shape))
                        batch_losses_temporal.append(np.mean(losses_temporal))
                    
                    elif retrain_mode == 'batch_pred':
                        with torch.no_grad():
                            x_list = []
                            for i in range(batch_size):
                                x_list.append(x)
                                y_true = targets[i, :, :].unsqueeze(0).to(self.device)
                                self.model.eval()
                                pred = self.model(x)[:, 0, :].view(1, 1, self.args.out_features)
                                if use_actual_actions:
                                    if self.args.out_features < self.args.in_features - self.num_time_f:
                                        if use_actual_ind:
                                            pred = torch.cat((y_true[:, :, :self.n_actions+self.n_ind], pred), dim=2)
                                        else:
                                            pred = torch.cat((y_true[:, :, :self.n_actions], pred), dim=2)
                                    else:
                                        if use_actual_ind:
                                            pred[:, :, :self.n_actions+self.n_ind] = y_true[:, :, :self.n_actions+self.n_ind] 
                                        else:
                                            pred[:, :, :self.n_actions] = y_true[:, :, :self.n_actions] 
                                
                                pred = torch.cat((pred, y_true[:, 0, self.args.in_features-self.num_time_f:].unsqueeze(0)), dim=2)
                                x = torch.cat([x, pred], dim=1).to(self.device)
                                x = x[:, 1:, :]
                        
                        x = torch.stack(x_list).view([batch_size, -1, self.args.in_features]).to(self.device)
                        targets = targets[:, :, self.args.in_features-self.num_time_f-self.args.out_features:
                                          self.args.in_features-self.num_time_f].to(self.device)
                        #self.retrain_dict['x'].append(np.array(x.detach().cpu()))
                        #self.retrain_dict['targets'].append(np.array(targets.detach().cpu()))

                        if loss_type == 'mse':
                            loss = self.train_step(x, targets)
                        if loss_type == 'dilate':
                            loss, loss_shape, loss_temporal = self.train_step(x, targets)
                            batch_losses_shape.append(loss_shape)
                            batch_losses_temporal.append(loss_temporal)
                        batch_losses.append(loss)
              
            val_loss, val_loss_dtw, val_loss_tdi = self.val_retrain(self.model, val_loader, use_actual_actions, use_actual_ind)
            train_loss = np.mean(batch_losses)
            train_loss_shape = np.mean(batch_losses_shape)
            train_loss_temporal = np.mean(batch_losses_temporal)
            self.train_losses.append(train_loss)
            self.train_losses_shape.append(train_loss_shape)
            self.train_losses_temporal.append(train_loss_temporal)

            self.val_losses.append(val_loss)
            self.val_losses_dtw.append(val_loss_dtw)
            self.val_losses_tdi.append(val_loss_tdi)
            
            if val_loss_type == 'dtw':
                val_loss = val_loss_dtw
            elif val_loss_type == 'tdi':
                val_loss = val_loss_tdi
            
            if best_loss is None or best_loss > val_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_path + '/' + retrain_name + '.pth')
                print('Simulation Testing ...')
                self.initiator.test_retrain()
                
                # print(f'Ex {experiment}: [Batch {i+1}/{len(retrain_data)}] | Idx: {batch_idx} | EL: {batch_targets.shape[1]}'+\
                      # f' | Loss: {batch_loss:.7f} | Best loss: {best_loss:.7f} | Train loss : {train_loss:.7f}')
                   
            print(f'### Ex {experiment}: [Epoch {epoch}/{epochs}] | Loss: {val_loss:.7f}  '+\
                  f'| Best loss: {best_loss:.7f} | Retrain loss : {train_loss:.7f} ###')
            
            # Logging to TensorBoard
            # writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, i)
            # writer.add_scalar('Best Val Loss', best_loss, i)
            
            if do_early_stop:
                early_stopping(val_loss, self.model, checkpoint_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
            gc.collect()
                
        self.end_timer_and_print(f"Mixed precision: {self.retrain_args.use_amp}")
        gc.collect()
        
        return
    
    def retrain_sepp(self, retrain_args, model, retrain_name, retrain_data, retrain_targets, 
                data_idxs, val_loader, test_loader, writer=None, sepp_gap=False):
        
        self.retrain_args = retrain_args
        checkpoints = self.args.checkpoints
        checkpoint_path = os.path.join(checkpoints, self.setting)
        epochs = self.retrain_args.retrain_epochs
        max_el = self.retrain_args.max_episode_length if (self.retrain_args.experiment == 2 or self.retrain_args.experiment == 4) else \
            self.retrain_args.const_episode_length
        use_actual_actions = self.retrain_args.use_actual_actions
        use_actual_ind = self.retrain_args.use_actual_ind
        loss_type = self.retrain_args.loss_function
        val_loss_type = self.retrain_args.val_loss_type
        patience = self.retrain_args.patience
        experiment = self.retrain_args.experiment
        do_early_stop = self.retrain_args.do_early_stop
        
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
           
        self.retrain_dict = {'x':[],
                            'targets':[],
                            'val_x':[],
                            'y_val':[],
                            'y_pred':[],
                            'val_losses':[],
                            'episode_losses':[]}
            
        self.model = model
        best_loss = None
        batch_size = 16
        
        print(f'Number of batches for each epoch: {len(retrain_data)}')
        
        # Making an instance of EarlyStopping class
        early_stopping = EarlyStoppingRetrain(patience=patience, verbose=True)
        
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.retrain_args.use_amp)
        self.start_timer()
        
        # Start retraining
        for epoch in range(1, epochs+1):
            batch_losses = []
            batch_losses_shape = []
            batch_losses_temporal = []

            for i, (data_item, target_item) in enumerate(tqdm(zip(retrain_data, retrain_targets))):
                # batch_idx = data_idxs[i] if experiment == 3 or experiment == 4 else i
                batch_data = torch.Tensor(np.array(data_item)).unsqueeze(0).to(self.device)
                batch_targets = torch.Tensor(np.array(target_item)).unsqueeze(0).to(self.device)
                
                retrain_dataset = TensorDataset(batch_data, batch_targets)
                retrain_loader = DataLoader(retrain_dataset, batch_size=1, 
                                          shuffle=False, drop_last=False)
                   
                for x, targets in retrain_loader:
                    x = x.to(self.device)
                    targets = targets.to(self.device)
                    batch_size = targets.shape[1] - self.args.pred_len + 1
                    targets_3d = torch.empty(batch_size, self.args.pred_len, self.args.in_features)
                    for i in range(batch_size):
                        targets_3d[i, :, :] = targets[:, i:i+self.args.pred_len, :]
                        
                    targets = targets_3d
                    #self.retrain_dict['targets_3d'].append(np.array(targets_3d.detach().cpu()))
                    pred = None                    
                    
                    episode_losses = []
                    episode_losses_shape = []
                    episode_losses_temporal = []

                    for el in range(1, batch_size+1):
                        with torch.no_grad():
                            x_list = []
                            for i in range(el):
                                x_list.append(x)
                                y_true = targets[i, :, :].unsqueeze(0).to(self.device)
                                self.model.eval()
                                pred = self.model(x)[:, 0, :].view(1, 1, self.args.out_features)
                                if use_actual_actions:
                                    if self.args.out_features < self.args.in_features - self.num_time_f:
                                        if use_actual_ind:
                                            pred = torch.cat((y_true[:, :, :self.n_actions+self.n_ind], pred), dim=2)
                                        else:
                                            pred = torch.cat((y_true[:, :, :self.n_actions], pred), dim=2)
                                    else:
                                        if use_actual_ind:
                                            pred[:, :, :self.n_actions+self.n_ind] = y_true[:, :, :self.n_actions+self.n_ind] 
                                        else:
                                            pred[:, :, :self.n_actions] = y_true[:, :, :self.n_actions] 
                                
                                pred = torch.cat((pred, y_true[:, 0, self.args.in_features-self.num_time_f:].unsqueeze(0)), dim=2)
                                x = torch.cat([x, pred], dim=1).to(self.device)
                                x = x[:, 1:, :]
                        
                        x_temp = torch.stack(x_list).view([el, -1, self.args.in_features]).to(self.device)
                        y_temp = targets[:el, :, self.args.in_features-self.num_time_f-self.args.out_features:
                                          self.args.in_features-self.num_time_f].to(self.device)
                        #self.retrain_dict['x'].append(np.array(x_temp.detach().cpu()))
                        #self.retrain_dict['targets'].append(np.array(y_temp.detach().cpu()))
    
                        if loss_type == 'mse':
                            loss = self.train_step(x_temp, y_temp)
                        if loss_type == 'dilate':
                            loss, loss_shape, loss_temporal = self.train_step(x_temp, y_temp)
                            episode_losses_shape.append(loss_shape)
                            episode_losses_temporal.append(loss_temporal)
                        
                        episode_losses.append(loss)
                        
                loss = np.mean(episode_losses)
                loss_shape = np.mean(episode_losses_shape)
                loss_temporal = np.mean(episode_losses_temporal)

                batch_losses.append(loss)
                batch_losses_shape.append(loss_shape)
                batch_losses_temporal.append(loss_temporal)
              
            val_loss, val_loss_dtw, val_loss_tdi = self.val_retrain(self.model, val_loader, use_actual_actions, use_actual_ind)
            train_loss = np.mean(batch_losses)
            train_loss_shape = np.mean(batch_losses_shape)
            train_loss_temporal = np.mean(batch_losses_temporal)
            self.train_losses.append(train_loss)
            self.train_losses_shape.append(train_loss_shape)
            self.train_losses_temporal.append(train_loss_temporal)

            self.val_losses.append(val_loss)
            self.val_losses_dtw.append(val_loss_dtw)
            self.val_losses_tdi.append(val_loss_tdi)
            
            if val_loss_type == 'dtw':
                val_loss = val_loss_dtw
            elif val_loss_type == 'tdi':
                val_loss = val_loss_tdi
            
            if best_loss is None or best_loss > val_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_path + '/' + retrain_name + '.pth')
                print('Simulation Testing ...')
                self.initiator.test_retrain()
                
                # print(f'Ex {experiment}: [Batch {i+1}/{len(retrain_data)}] | Idx: {batch_idx} | EL: {batch_targets.shape[1]}'+\
                      # f' | Loss: {batch_loss:.7f} | Best loss: {best_loss:.7f} | Train loss : {train_loss:.7f}')
                   
            print(f'### Ex {experiment}: [Epoch {epoch}/{epochs}] | Loss: {val_loss:.7f}  '+\
                  f'| Best loss: {best_loss:.7f} | Retrain loss : {train_loss:.7f} ###')
            
            # Logging to TensorBoard
            # writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, i)
            # writer.add_scalar('Best Val Loss', best_loss, i)
            
            if do_early_stop:
                early_stopping(val_loss, self.model, checkpoint_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
            gc.collect()

        gc.collect()
        return
    
    # Validation of the retrain results
    def val_retrain(self, model, val_loader, use_actual_actions, use_actual_ind):
        """The method validates the retrained model

        The method takes retrained model, DataLoaders for the val dataset, and
        whether to use actual actions or not. It iteratively
        predicts the target values and calculates losses and return them.

        Args:
            model : The loaded model from the checkpoint  
            val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
            use_actual_actions (boolean): Whether to replace actual action values at each step of the prediction

        Returns:
            losses: mse, dtw, tdi

        """
        # Start validation
        with torch.no_grad():
            batch_val_losses = []
            batch_val_losses_dtw = []
            batch_val_losses_tdi = []
            for x_val, y_val in val_loader:
                val_length = y_val.shape[1] - self.args.pred_len + 1
                y_val_3d = torch.empty(val_length, self.args.pred_len, self.args.in_features)
                for i in range(val_length):
                    y_val_3d[i, :, :] = y_val[:, i:i+self.args.pred_len, :]
                pred_list = []
                x_list = []
                x = x_val
                for i in range(val_length):
                    x_list.append(x)
                    y_true = y_val_3d[i, :, :].unsqueeze(0).to(self.device)
                    self.model.eval()
                    pred = self.model(x)[:, 0, :].view(1, 1, self.args.out_features)
                    if use_actual_actions:
                        if self.args.out_features < self.args.in_features - self.num_time_f:
                            if use_actual_ind:
                                pred = torch.cat((y_true[:, 0, :self.n_actions+self.n_ind].unsqueeze(0), pred), dim=2)
                                pred_list.append(pred[:, :, self.n_actions+self.n_ind:self.args.in_features-self.num_time_f])
                            else:
                                pred = torch.cat((y_true[:, 0, :self.n_actions].unsqueeze(0), pred), dim=2)
                                pred_list.append(pred[:, :, self.n_actions:self.args.in_features-self.num_time_f])
                        else:
                            if use_actual_ind:
                                pred[:, :, :self.n_actions+self.n_ind] = y_true[:, :, :self.n_actions+self.n_ind] 
                                pred_list.append(pred[:, :, self.args.out_features])
                            else:
                                pred[:, :, :self.n_actions] = y_true[:, :, :self.n_actions] 
                                pred_list.append(pred[:, :, self.args.out_features])
                    
                    pred = torch.cat((pred, y_true[:, 0, self.args.in_features-self.num_time_f:].unsqueeze(0)), dim=2)
                    x = torch.cat([x, pred], dim=1)[:, 1:, :].to(self.device)
                
                y_pred = torch.stack(pred_list).view([1, val_length, self.args.out_features]).to(self.device)
                if self.args.out_features < self.args.in_features - self.num_time_f:
                    y_val = y_val[:, :val_length, self.n_actions+self.n_ind:self.args.in_features-self.num_time_f].to(self.device)
                else:
                    y_val = y_val[:, :val_length, :self.args.in_features-self.num_time_f].to(self.device)
                    
                # self.retrain_dict['val_x'].append(np.array(x.detach().cpu()))
                # self.retrain_dict['y_val'].append(np.array(y_val.detach().cpu()))
                # self.retrain_dict['y_pred'].append(np.array(y_pred.detach().cpu()))
                val_loss = self.loss_fn(y_val.detach().cpu(), y_pred.detach().cpu()).item()
                batch_val_losses.append(val_loss)
                
                # DTW and TDI
                val_loss_dtw, val_loss_tdi = 0, 0
                for k in range(y_val.shape[0]):   
                    batch_size, N_output = y_val[:, :, self.target_idx].view([1, val_length, 1]).shape[0:2]
                    target = y_val[:, :, self.target_idx].view([1, val_length, 1])
                    outputs = y_pred[:, :, self.target_idx].view([1, val_length, 1])
                    target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
                    output_k_cpu = outputs[k,:,0:1].view(-1).detach().cpu().numpy()

                    path, sim = dtw_path(target_k_cpu, output_k_cpu)   
                    val_loss_dtw += sim
                               
                    Dist = 0
                    for i,j in path:
                            Dist += (i-j)*(i-j)
                    val_loss_tdi += Dist / (N_output*N_output)            
                                
                val_loss_dtw = val_loss_dtw / batch_size
                val_loss_tdi = val_loss_tdi / batch_size
                batch_val_losses.append(val_loss)
                batch_val_losses_dtw.append(val_loss_dtw)
                batch_val_losses_tdi.append(val_loss_tdi)
            validation_loss = np.mean(batch_val_losses)
            validation_loss_dtw = np.mean(batch_val_losses_dtw)
            validation_loss_tdi = np.mean(batch_val_losses_tdi)
            self.val_losses.append(validation_loss)
            self.val_losses_dtw.append(validation_loss_dtw)
            self.val_losses_tdi.append(validation_loss_tdi)
            gc.collect()
                    
            # self.retrain_dict['val_losses'].append(np.array(batch_val_losses))
            # return abs(validation_loss), abs(validation_loss_dtw), abs(validation_loss_tdi)
            return validation_loss, validation_loss_dtw, validation_loss_tdi
        
    def test_simulation(self, model, test_loader):
        """
        Run the simulation environment using a saved model and test data loader.
    
        Args:
            model (torch.nn.Module): The loaded model from the checkpoint.
            test_loader (torch.utils.data.DataLoader): DataLoader that stores simulation test data.
            ctrl_sep (bool, optional): Control separator flag, default is False.
            ctrl_vals (optional): Control values, default is None.
    
        Returns:
            list: List of dictionaries containing test results for each batch.
            list: List of losses for each batch during simulation.
        """
        self.model = model
        test_results_list = []
        batch_test_losses = []

        with torch.no_grad():
            for x_test, y_test in tqdm(test_loader):
                test_results_dict = {}
                test_length = y_test.shape[1] + 1 - self.args.pred_len
                pred_list = []
                x_list = []

                x = x_test.to(self.device)

                for i in range(test_length):
                    x_list.append(x)
                    y_true = y_test[:, i:i+self.args.pred_len, :].to(self.device)
                    
                    self.model.eval()
                    pred = self.model(x)
                    
                    out_features = pred.shape[-1]
                    
                    if self.is_policy:
                        pred = torch.cat((pred, y_true[:, :, self.n_actions:-self.num_time_f]), dim=2)
                    else:
                        pred = torch.cat((y_true[:, :, :self.args.in_features - self.num_time_f - out_features], pred), dim=2)
                    
                    pred_list.append(pred[:, 0, :])
                    
                    pred = torch.cat((pred, y_true[:, :, self.args.in_features - self.num_time_f:]), dim=2)
                    x = torch.cat([x, pred[:, 0, :].unsqueeze(0)], dim=1).to(self.device)
                    x = x[:, 1:, :]
                
                y_pred = torch.stack(pred_list).view([test_length, self.args.in_features - self.num_time_f]).to(self.device)
                y_real = y_test[:, :test_length, :self.args.in_features - self.num_time_f].squeeze(0).to(self.device)
                
                test_loss = self.loss_fn(y_real.cpu(), y_pred.cpu()).item()
                
                # Compute DTW and TDI losses
                loss_dtw, loss_tdi = 0, 0
                for k in range(y_test.shape[0]):
                    target = y_real[:, self.target_idx].unsqueeze(0).view([1, test_length, 1])
                    outputs = y_pred[:, self.target_idx].unsqueeze(0).view([1, test_length, 1])
                    
                    target_k_cpu = target[k, :, 0:1].view(-1).cpu().numpy()
                    output_k_cpu = outputs[k, :, 0:1].view(-1).cpu().numpy()
                    
                    path, sim = dtw_path(target_k_cpu, output_k_cpu)
                    loss_dtw += sim
                    
                    dist = sum((i - j) ** 2 for i, j in path)
                    loss_tdi += dist / (test_length ** 2)
                
                loss_dtw /= y_test.shape[0]
                loss_tdi /= y_test.shape[0]
                
                # Collect results
                test_results_dict['y_real'] = y_real.cpu().numpy()
                test_results_dict['y_pred'] = y_pred.cpu().numpy()
                test_results_dict['test_loss'] = test_loss
                test_results_dict['loss_dtw'] = loss_dtw
                test_results_dict['loss_tdi'] = loss_tdi

                test_results_list.append(test_results_dict)
                batch_test_losses.append(test_loss)

        gc.collect()
        
        return test_results_list, batch_test_losses
    
    def improve_model_env(self, x, targets, retrain_epochs, use_actual_actions, improve_mode):
        best_model = None
        best_loss = None
        for epoch in range(1, retrain_epochs+1):
            if improve_mode == 'batch_pred':
                batch_size = targets.shape[1]
                x_list = []
                y_list = []
                for i in range(batch_size):
                    x_list.append(x)
                    y_true = targets[:,i,:].to(self.device)
                    y_list.append(y_true[:, :self.args.out_features])
                    # print('y_true shape: ', y_true.shape)
                    with torch.no_grad():
                        self.model.eval()
                        pred = self.model(x)[:, 0, :].view(1, self.args.out_features)
            
                        if use_actual_actions:
                            pred[:, self.n_actions+self.n_ind] = y_true[:, self.n_actions+self.n_ind]
                        
                        pred = torch.cat((pred, y_true[:, self.args.out_features:]), dim=1)
                        # print('prev_pred shape: ', prev_pred.shape)
                        x = torch.cat([x, pred.unsqueeze(0)], dim=1).to(self.device)
                        x = x[:, 1:, :]
                        
                x_final = torch.stack(x_list).view([batch_size, -1, self.args.in_features]).to(self.device)
                y_final = torch.stack(y_list).view([batch_size, -1, self.args.out_features]).to(self.device)     
                epoch_loss = self.train_step(x_final, y_final)
            
            elif improve_mode == 'single_pred':
                prev_pred = None
                sum_losses = 0
                losses = []
                for i in range(targets.shape[1]):
                    y_true = targets[:,i,:].to(self.device)
                    if prev_pred is not None:
                        prev_pred = torch.cat((prev_pred, y_true[:, self.args.out_features:]), dim=1)
                        # print('prev_pred shape: ', prev_pred.shape)
                        x = torch.cat([x, prev_pred.unsqueeze(0)], dim=1).to(self.device)
                        x = x[:, 1:, :]
                    # print('second x shape: ', x.shape)
                    # print('y pred shape: ', y_pred.shape)
                    # print('y true shape: ', y_true.shape)
                    loss = self.train_step(x, y_true[:,:self.args.out_features].unsqueeze(0))                  
                    losses.append(loss)    
                    sum_losses += loss
                    
                    with torch.no_grad():
                        self.model.eval()
                        prev_pred = self.model(x)[:, 0, :].view(1, self.args.out_features)
                        # print('prev_pred: ', prev_pred.shape)
                        if use_actual_actions:
                            prev_pred[:, 0] = y_true[:, 0]
                                    
                # Losses of the whole episode
                epoch_loss = np.mean(losses)
                
            if epoch == 1 or epoch % 5 == 0:
                print(f'[Epoch {epoch}/{retrain_epochs}] | Loss: {epoch_loss:.7f} ')
                
            if best_loss == None or best_loss > epoch_loss:
                best_model = self.model
                
        return best_model
    
    # To sample actions when is_policy = True
    def sample_normal():
        pass
