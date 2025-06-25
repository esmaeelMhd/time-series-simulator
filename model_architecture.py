import torch
from torchviz import make_dot
from models.LSTM import LSTMModel
from utils.LSTM_model_optimizer import Optimization
import warnings
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

setting = 'LSTM_New_CorrH_11F_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256Hidden_2LayerDim'
ARGS_PATH = './args/' + setting + '/'
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    with open(ARGS_PATH + 'args.pkl', 'rb') as file:
        args = pickle.load(file)

model = LSTMModel(args, device).float()

# Assuming model is your LSTM model and x is a dummy input tensor
x = torch.randn(1, args.seq_len, args.in_features)  # Replace seq_len and num_features with appropriate values
y = model(x)

dot = make_dot(y, params=dict(list(model.named_parameters())))
dot.render("lstm_architecture", format="png")
