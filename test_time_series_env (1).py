from envs.time_series_env import DataArgs, AgentArgs, TimeSeriesEnv
from envs.phosphorus_reward import PhosphorusReward
from envs.utils import load_dataset, load_args, args_mapping, ModelBuilder, ScalerHandler

"""Data information:"""
DATA_ROOT_PATH='./datasets'
data_name='wastewater.csv'
### data variables information
ctrl_vars=['IN_METAL_Q', 'T1_O2', 'METAL_Q']
ind_vars=['TEMPERATURE', 'IN_Q', 'MAX_CF', 'PROCESSPHASE_INLET', 'PROCESSPHASE_OUTLET']
target_vars='T1_PO4'
obs_vars=['T1_PO4']

"""Model information:"""
ARGS_ROOT_PATH = './args'
SCALER_ROOT_PATH = './scalers'
model_type = 'LSTM'
model_name = 'LSTM_IOPTQCfFiFoP_2min_15F_1Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L'
checkpoint = 'E2N1 - 80Ep_SGap_480MaxEL_Batch_d_all_0.6A_1e-06LR_Vdtw.pth'

args = load_args(args_path=ARGS_ROOT_PATH, model_name=model_name)

map_dict = {
    'root_path': 'data_root_path',
    'data_path': 'data_name',
    'model': 'model_type',
    'setting': 'model_name',
    'control_variable': 'ctrl_vars',
    'ctrl_vars': 'ctrl_vars',
    'independent_vars': 'ind_vars',
    'ind_vars': 'ind_vars',
    'target': 'target_vars',
    'target_vars': 'target_vars',
    'num_time_f': 'num_time_f',
    'time_scaled': 'time_scaled',
    'embed': 'time_f'
    }
args = args_mapping(args, map_dict)

model_builder = ModelBuilder(**vars(args))

model = model_builder.load_model()
exp = model_builder.initialize_model_exp()
df = load_dataset(data_root_path=DATA_ROOT_PATH, data_name=data_name)
scaler_handler = ScalerHandler(df=df, model_name=model_name)

stop

model_args = DataArgs(
    data_root_path=DATA_ROOT_PATH, 
    data_name=data_name, 
    model_type=model_type, 
    model_name=model_name, 
    checkpoint=checkpoint, 
    scale=True, 
    ctrl_vars=ctrl_vars, 
    ind_vars=ind_vars, 
    target_vars=target_vars,
    obs_vars=obs_vars,
    time_f=True,
    num_time_f=6,
    time_scaled='unscaled')

agent_args = AgentArgs(
    agent_name='Agent', 
    experiment=1,
    const_el=360,
    delay_type='constant',
    const_delay=90)

env = TimeSeriesEnv(
    model_args, 
    agent_args, 
    reward_function=PhosphorusReward(target='T1_PO4', q_column='IN_Q'), 
    results_folder='./agent_results')