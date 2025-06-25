import time
import numpy as np
from tqdm import tqdm

from envs.time_series_env import ModelArgs, AgentArgs, TimeSeriesEnv
from envs.phosphorus_reward import PhosphorusReward
from envs.utils import load_dataset, load_args, args_mapping, ModelBuilder, ScalerHandler
from env_wrapper import RandomDelayWrapper

def main():
    """Data information:"""
    data_root_path='./datasets'
    data_name='wastewater.csv'
    scale = True
    index_col = 'date'
    time_f = True
    has_time_f = False
    num_time_f = 6
    time_scaled = 'unscaled'
    
    ### data variables information
    ctrl_vars=['IN_METAL_Q', 'T1_O2', 'METAL_Q']
    ind_vars=['TEMPERATURE', 'IN_Q', 'MAX_CF', 'PROCESSPHASE_INLET', 'PROCESSPHASE_OUTLET']
    target_vars='T1_PO4'
    obs_vars=['T1_PO4']
    
    """Model information:"""
    model_args_path = './args'
    scaler_root_path = './scalers'
    model_type = 'LSTM'
    model_name = 'LSTM_IOPTQCfFiFoP_2min_15F_1Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L'
    checkpoint = 'E2N1 - 80Ep_SGap_480MaxEL_Batch_d_all_0.6A_1e-06LR_Vdtw.pth'
    
    """Load the dataset:"""
    df = load_dataset(data_root_path=data_root_path, data_name=data_name)
    
    """Load saved model args:"""
    model_args = load_args(model_args_path, model_name)
    
    # Map the saved args attributes to new attribute names
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
    model_args = args_mapping(model_args, map_dict)
    
    """Build the model:"""
    model_builder = ModelBuilder(args=model_args, device='cpu', model_type=model_type,
                                 model_name=model_name, checkpoint=checkpoint)    
    model = model_builder.load_model()
    exp = model_builder.initialize_model_exp()
    
    """Build the scaler handler:"""
    scaler_handler = ScalerHandler(args=model_args, scaler_root_path=scaler_root_path, 
                                   df=df, model_name=model_name)  
    
    """Agent information:"""
    agent_args_path = './agent_args'
    agent_name = 'Test_agent_time_series_env'
    experiment = 1
    const_el = 60
    min_el = 10
    max_el = 480
    delay_type = 'random'
    const_delay = 10
    act_delay_max = 1
    obs_delay_max = 4
    title: str = 'Test Agent'
    norm_values = True
    
    """Load saved agent args:"""
    # agent_args = load_args(agent_args_path, agent_name)
    agent_args = None
    
    """Step 1: build the model args:"""
    model_args = ModelArgs.from_namespace(
        args=model_args,
        data_root_path=data_root_path, 
        data_name=data_name, 
        model_type=model_type, 
        model_name=model_name, 
        checkpoint=checkpoint, 
        scale=scale, 
        index_col=index_col,
        time_f=time_f,
        has_time_f=has_time_f,
        num_time_f=num_time_f,
        time_scaled=time_scaled,
        ctrl_vars=ctrl_vars, 
        ind_vars=ind_vars, 
        target_vars=target_vars,
        obs_vars=obs_vars
        )
    
    """Step 2: build the agent/controller args:"""
    agent_args = AgentArgs.from_namespace(
        args=agent_args,
        agent_name=agent_name, 
        experiment=experiment,
        const_el=const_el,
        min_el=min_el,
        max_el=max_el,
        delay_type=delay_type,
        const_delay=const_delay,
        title=title,
        norm_values=norm_values,
        )
    
    """Step 3: bulid the environment"""
    env = TimeSeriesEnv(
        model_args, 
        agent_args,
        model=model,
        exp=exp,
        scaler_handler=scaler_handler,
        device='cpu',
        reward_function=PhosphorusReward(target_vars=target_vars, target='T1_PO4', q_column='IN_Q'), 
        df_raw=df,
        results_folder='./agent_results')
    
    """testing the environment:"""
    z_obs = env.observations
    z_target = env.targets
    z_actions = env.actions
    z_sequence = env.sequences
    z_rewards = env.rewards
    iterations = 1
    actions = np.random.random((const_el, len(ctrl_vars)))
    print('Testing the environment ...')
    # iterations is the same as epochs
    for i in tqdm(range(iterations)):
        steps = 0
        start_time = time.time()
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = actions[steps]
            observation_, reward, done, info = env.step(action)
            steps += 1
            score += reward
            observation = observation_
            
        end_time = time.time()
    
    print(f'Testing time: {(end_time-start_time)/60:.2f} min')
    
    """Build the environment wrapper for random delay:"""
    if agent_args.delay_type == 'random':
        env = RandomDelayWrapper(env=env, 
                                 initial_action=np.array([0.1, 0.1, 0.1]), 
                                 obs_delay_range=range(0, obs_delay_max),
                                 act_delay_range=range(0, act_delay_max))
        env.reset()
    
    print('Testing the environment for random delay wrapper ...')
    def test_env_wrapper(env):
        action = np.array([0.5, 0.5, 0.5])
        list_past_actions = []
        list_past_observations = []
        list_arrival_times_actions = []
        list_arrival_times_observations = []
        list_cum_rew_brain = []
        list_next_action = []
        list_rewards = []
        list_obs = []
        for i in range(20):
            obs, reward, done, info = env.step(action)
            list_past_actions.append(list(env.past_actions))
            list_past_observations.append(list(env.past_observations))
            list_arrival_times_actions.append(list(env.arrival_times_actions))
            list_arrival_times_observations.append(list(env.arrival_times_observations))
            list_cum_rew_brain.append(env.cum_rew_brain)
            list_next_action.append(env.next_action)
            list_rewards.append(reward)
            list_obs.append(obs)
            action = action + 0.1
    
    test_env_wrapper(env) 

if __name__ == '__main__':
    main()