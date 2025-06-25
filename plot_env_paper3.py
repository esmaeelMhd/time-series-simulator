import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def load_predictions(model_folder):
    """
    Load all prediction results from pickle files in the specified model folder.
    """
    checkpoints = {}
    for file in os.listdir(os.path.join('./env_results', model_folder)):
        if file.endswith('.pkl'):
            with open(os.path.join('./env_results', model_folder, file), 'rb') as f:
                predictions = pickle.load(f)
                checkpoints[file] = predictions
    return checkpoints

def evaluate_best_checkpoint(checkpoints, loss_type='mse'):
    """
    Evaluate the best checkpoint based on the specified loss metric (mse or dtw).
    """
    best_metric = float('inf')
    best_checkpoint = None
    
    for checkpoint, results in checkpoints.items():
        total_loss = 0
        for date, data in results.items():
            if loss_type == 'mse':
                loss = mean_squared_error(data['y_real'], data['y_pred'])
            elif loss_type == 'dtw':
                loss = data['loss_dtw']
            total_loss += loss
        
        avg_loss = total_loss / len(results)
        if avg_loss < best_metric:
            best_metric = avg_loss
            best_checkpoint = checkpoint
    
    return best_checkpoint, checkpoints[best_checkpoint]

def plot_losses(models, loss_type='mse', save_path='./env_results/All Best/'):
    """
    Plot the loss throughout all date points for the best checkpoints of all models.
    """
    plt.figure(figsize=(12, 8))
    
    for model_folder in models:
        data_tag, in_features, out_features = extract_model_info(model_folder)
        checkpoints = load_predictions(model_folder)
        best_checkpoint, best_predictions = evaluate_best_checkpoint(checkpoints, loss_type)
        
        dates = []
        losses = []
        for date, data in best_predictions.items():
            dates.append(date)
            if loss_type == 'mse':
                loss = mean_squared_error(data['y_real'], data['y_pred'])
            elif loss_type == 'dtw':
                loss = data['loss_dtw']
            losses.append(loss)
        
        plt.plot(dates, losses, label=f'{data_tag} - {in_features} in {out_features} out')
    
    plt.xlabel('Date')
    plt.ylabel('Loss')
    plt.title('Loss Throughout All Date Points for Best Checkpoints')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_comparison.png'))
    plt.show()

def plot_predictions(models, specific_date, save_path='./env_results/All Best/'):
    """
    Plot the predictions and the real data for a specific date point for all models.
    """
    plt.figure(figsize=(12, 8))
    
    for model_folder in models:
        data_tag, in_features, out_features = extract_model_info(model_folder)
        checkpoints = load_predictions(model_folder)
        best_checkpoint, best_predictions = evaluate_best_checkpoint(checkpoints)
        
        if specific_date in best_predictions:
            y_real = best_predictions[specific_date]['y_real']
            y_pred = best_predictions[specific_date]['y_pred']
            plt.plot(y_real, label=f'{data_tag} - {in_features} in {out_features} out - Real')
            plt.plot(y_pred, label=f'{data_tag} - {in_features} in {out_features} out - Pred')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Predictions vs Real Data for {specific_date}')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'predictions_{specific_date}.png'))
    plt.show()

def extract_model_info(model_folder):
    """
    Extract dataset name, in features, and out features from the model folder name.
    """
    parts = model_folder.split('_')
    data_tag = parts[1]
    in_features = parts[3].replace('F', '')
    out_features = parts[4].replace('Out', '')
    return data_tag, in_features, out_features

# Example usage:
model_folders =[# 'LSTM_IOPP_2min_10F_1Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L',
                     'LSTM_IOPTQCfFiFoP_2min_15F_1Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L',
                     'LSTM_IOPTQCfFiFoP_2min_15F_6Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L']

specific_date = '2022-09-15'  # Example specific date

# Ensure the save path exists
save_path = './env_results/All Best/'
os.makedirs(save_path, exist_ok=True)

# Plot losses for best checkpoints of all models
plot_losses(model_folders, loss_type='mse', save_path=save_path)

# Plot predictions vs real data for a specific date
plot_predictions(model_folders, specific_date, save_path=save_path)
