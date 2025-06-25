"""
Created on Monday Dec 01 2023
@author: Esmaeel Mohammadi

# =============================================================================
# This script is used to train and test Soft Actor-Critic (SAC) and 
  Delay Correcting Actor-Critic (DCAC) algorithm:
    1- Create an instance of environment
    2- Choose the method, based on needs:
        - SAC with no delay - Real Rime Reinforcement Learning (RTRL)
        - SAC with constant delay - History of actions and observation as input 
        to the actor network (obs dim = history * (num_actions + num_obs))
        - SAC with random and/or non-constant delay
        - DCAC with random delay (https://openreview.net/pdf?id=QFYnKlBJYR)
    3- Choose the netwoks type (Mlp - Delayed Mlp - LSTM)
    4- Train the agent on the environment
    5- Test the trained policy
# =============================================================================
"""
# Multi GPU:
# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
import argparse
import os
import pickle
import random
import time
import atexit
import gc
import json
import os
import shutil
import tempfile
from os.path import exists
from random import randrange
from tempfile import mkdtemp
import logging
# for the cloud
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import torch
import gym
from gym.envs.registration import register

from envs.time_series_env import ModelArgs, AgentArgs, TimeSeriesEnv
from envs.phosphorus_reward import PhosphorusReward
from envs.utils import load_dataset, load_args, args_mapping, ModelBuilder, ScalerHandler
from env_wrapper import RandomDelayWrapper

from agents import SACAgent, SACAgentRD, DCACAgent
from sac_utils import plot_learning_curve, plot_avg_learning_curve, EarlyStopping

from rlrd.envs import Env, GymEnv, RandomDelayEnv
import rlrd.sac_models
import rlrd.sac_models_rd
import rlrd.dcac_models
from rlrd.util import partial
from rlrd.nn import PopArt

import yaml

from rlrd.util import partial, save_json, partial_to_dict, partial_from_dict, load_json, dump, load, git_info
from rlrd.training import Training

#%%
def iterate_episodes(run_cls: type = Training, checkpoint_path: str = None):
    """Generator [1] yielding episode statistics (list of pd.Series) while running and checkpointing
    - run_cls: can by any callable that outputs an appropriate run object (e.g. has a 'run_epoch' method)

    [1] https://docs.python.org/3/howto/functional.html#generators
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

    try:
        if not exists(checkpoint_path):
            print("=== specification ".ljust(70, "="))
            print(yaml.dump(partial_to_dict(run_cls), indent=3, default_flow_style=False, sort_keys=False), end="")
            run_instance = run_cls()
            dump(run_instance, checkpoint_path)
            print("")
        else:
            print("\ncontinuing...\n")

        run_instance = load(checkpoint_path)
        while run_instance.epoch < run_instance.epochs:
            # time.sleep(1)  # on network file systems writing files is asynchronous and we need to wait for sync
            yield run_instance.run_epoch()  # yield stats data frame (this makes this function a generator)
            print("")
            dump(run_instance, checkpoint_path)

            # we delete and reload the run_instance from disk to ensure the exact same code runs regardless of interruptions
            del run_instance
            gc.collect()
            run_instance = load(checkpoint_path)

    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
            os.remove(checkpoint_path)


def log_environment_variables():
    """add certain relevant environment variables to our config
    usage: `LOG_VARIABLES='HOME JOBID' python ...`
    """
    return {k: os.environ.get(k, '') for k in os.environ.get('LOG_VARIABLES', '').strip().split()}


def run(run_cls: type = Training, checkpoint_path: str = None):
    list(iterate_episodes(run_cls, checkpoint_path))

def run_wandb(entity, project, run_id, run_cls: type = Training, checkpoint_path: str = None):
    """run and save config and stats to https://wandb.com"""
    wandb_dir = mkdtemp()  # prevent wandb from polluting the home directory
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
    import wandb
    config = partial_to_dict(run_cls)
    config['seed'] = config['seed'] or randrange(1, 1000000)  # if seed == 0 replace with random
    config['environ'] = log_environment_variables()
    config['git'] = git_info()
    resume = checkpoint_path and exists(checkpoint_path)
    wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)
    for stats in iterate_episodes(run_cls, checkpoint_path):
        [wandb.log(json.loads(s.to_json())) for s in stats]


def run_fs(path: str, run_cls: type = Training):
    """run and save config and stats to `path` (with pickle)"""
    if not exists(path):
        os.mkdir(path)
    save_json(partial_to_dict(run_cls), path + '/spec.json')
    if not exists(path + '/stats'):
        dump(pd.DataFrame(), path + '/stats')
    for stats in iterate_episodes(run_cls, path + '/state'):
        dump(load(path + '/stats').append(stats, ignore_index=True),
             path + '/stats')  # concat with stats from previous episodes
#%%
def make_args():
    parser = argparse.ArgumentParser(description='Configure experiment parameters for delayed RL training')

    # Define arguments with defaults
    parser.add_argument('--agent_type', type=str, default=None, help='the type of the agent')
    parser.add_argument('--policy', type=str, default=None, help='the policy method')
    parser.add_argument('--title', type=str, default=None, help='title for training the agent')
    parser.add_argument('--norm_values', type=bool, default=None, help='either to scale the obs, rewards, ...')
    parser.add_argument('--experiment', type=int, default=None, help='the experiment: 1, 2, 3, 4')
    parser.add_argument('--const_el', type=int, default=None, help='the constant episode length')
    parser.add_argument('--min_el', type=int, default=None, help='the min episode length')
    parser.add_argument('--max_el', type=int, default=None, help='the max episode length')
    parser.add_argument('--delay_type', type=str, default=None, help='the delay type: constant or random')
    parser.add_argument('--const_delay', type=int, default=None, help='the max delay of the system')
    parser.add_argument('--act_delay_max', type=int, default=None, help='the max delay of the actions')
    parser.add_argument('--obs_delay_max', type=int, default=None, help='the max delay of the observations')
    parser.add_argument('--env_id', type=str, default=None, help='the id of the environment')
    parser.add_argument('--agent_name', type=str, default=None, help='the name of the agent')
    parser.add_argument('--early_stopping', type=bool, default=None, help='perform early stopping or not')
    parser.add_argument('--patience', type=int, default=None, help='the patience for early stopping')
    parser.add_argument('--lr_actor', type=float, default=None, help='learning rate of the actor')
    parser.add_argument('--lr_critic', type=float, default=None, help='learning rate of the critic')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size for the training')
    parser.add_argument('--reward_type', type=str, default=None, help='the type of the reward function')
    parser.add_argument('--reward_scale', type=float, default=None, help='the scale for the reward')
    parser.add_argument('--tau', type=float, default=None, help='soft update parameter for the target networks')
    parser.add_argument('--actor_net', type=str, default=None, help='type of the actor network: LSTM, ...')
    parser.add_argument('--obs_history', type=int, default=None, help='the history of observation to be returned by the env')

    return parser.parse_args()

def set_defaults(args):     
    # Default values mapping
    defaults = {
        # agent specifications
        # agent_type: 1- sac_lstm, 2- sac_mlp_1, 3- sac_mlp_2, 4- dcac
        'agent_type': 'sac_mlp_2',
        'title': 'Random Delay',
        'actor_net': '',
        'policy':'',

        # delay specifications
        # delay_type: 1- none, 2- random, 3- constant
        'delay_type': 'random',
        'const_delay': 3,
        'act_delay_max': 3,
        'obs_delay_max': 5,
        
        # env specifications
        'env_id': 'P_const_delay',
        'norm_values': True,
        'obs_history': 1,
        'experiment': 1,
        'const_el': 10,
        'min_el': 90,
        'max_el': 1440,

        
        # reward specifications
        'reward_type': 'linear_pmt',
        'reward_scale': 2,
        'tau': 0.005,
        
        # training specifications
        'iterations': 20,
        'init_policy': 'LSTM_IOPTQCfP_13F_3Out_timeF_Unscaled_90Seq_16Batch_1e-06LR_80H_1L',
        'lr_actor': 0.0003,
        'lr_critic': 0.0003,
        'batch_size': 100,
        'early_stopping': True,
        'patience': 10
    }
    
    # Apply defaults if not provided in command line
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    
    if args.delay_type != 'constant' and args.actor_net.lower() == 'lstm':
            raise ValueError("The LSTM network is used only for constant delay")
            
    if args.agent_type != 'sac_lstm' and args.actor_net.lower() != 'lstm':
            args.actor_net == 'lstm'
    
    if args.delay_type == 'constant' and args.actor_net.lower() == 'lstm':
        args.obs_history = args.const_delay
        
    # Dynamic settings
    if args.patience is None:
        args.patience = int(np.ceil(0.1 * args.iterations))
    
    if args.agent_type == 'dcac' and args.delay_type != 'random':
        args.delay_type == 'random'
    
    # Calculate `agent_name` based on other parameters
    el = f'{args.max_el}MaxEL' if args.experiment in [2, 4] else f'{args.const_el}EL'
    if args.delay_type == 'none':
        delay = 'no_delay'
    elif args.delay_type == 'none':
        delay = 'const_delay'
    else:
        delay = 'random_delay'
    args.policy = args.agent_type + '_' + delay
    agent_name = f'{args.policy}_E{args.experiment}_{el}_{args.reward_type}_rw_{args.iterations}Itr'
    args.agent_name = agent_name

#%%
def main():
    """Preparation:"""
    torch.cuda.empty_cache()
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    """Data information:"""
    data_root_path = './datasets'
    data_name = 'wastewater.csv'
    scale = True
    index_col = 'date'
    time_f = True
    has_time_f = False
    num_time_f = 6
    time_scaled = 'unscaled'
    
    ### data variables information
    ctrl_vars = ['IN_METAL_Q', 'T1_O2', 'METAL_Q']
    ind_vars = ['TEMPERATURE', 'IN_Q', 'MAX_CF', 'PROCESSPHASE_INLET', 'PROCESSPHASE_OUTLET']
    target_vars = 'T1_PO4'
    obs_vars = ['T1_PO4']
    
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
    model_builder = ModelBuilder.from_namespace(args=model_args, device='cpu', model_type=model_type,
                                                model_name=model_name, checkpoint=checkpoint) 
    
    model = model_builder.load_model()
    exp = model_builder.initialize_model_exp()
    
    """Build the scaler handler:"""
    scaler_handler = ScalerHandler.from_namespace(args=model_args, scaler_root_path=scaler_root_path, 
                                                  df=df, model_name=model_name)
    
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
    args = make_args()
    set_defaults(args) 
    agent_results_folder = './agent_results'
    agent_args = AgentArgs.from_namespace(args, title='test change')
    
    """Create the reward function:"""
    phosphorus_reward = PhosphorusReward(target_vars=target_vars, target='T1_PO4', q_column='IN_Q')
    
    """Step 3: bulid the environment"""
    def create_time_series_env(agent_type, delay_type):
        """Creates the base TimeSeriesEnv environment"""
        return TimeSeriesEnv(
            model_args, 
            agent_args,
            model=model,
            exp=exp,
            scaler_handler=scaler_handler,
            device='cpu',
            reward_function=phosphorus_reward, 
            df_raw=df,
            results_folder=agent_results_folder
        )
    
    def create_environment(args):
        base_env = create_time_series_env(args.agent_type, args.delay_type)
    
        if args.agent_type in ['sac_lstm', 'sac_mlp_1']:
            return base_env
        elif args.agent_type == 'sac_mlp_2' and args.delay_type in  ['none', 'constant']:
            return GymEnv(base_env)
        else:
            return RandomDelayEnv(
                base_env,  # env
                min_observation_delay=0,
                sup_observation_delay=args.obs_delay_max,
                min_action_delay=0,
                sup_action_delay=args.act_delay_max
            )
    
    # Usage
    env = create_environment(args)
        
    """Step 4: bulid the agent"""
    if args.agent_type in ['sac_lstm', 'sac_mlp_1']:
        agent = SACAgent.from_namespace(
            args=args,
            env=env,
            input_dims=env.observation_space.shape,
            n_actions=env.action_space.shape[-1],
            max_mem_size=3000,
            layer1_size=256,
            layer2_size=256,
            batch_size=20
        )
    elif args.agent_type == 'sac_mlp_2':
        if args.delay_type == 'constant':
            agent = SACAgentRD.from_namespace(
                args=args,
                env=env,
                memory_size=3000,
                layer_size=256,
                batch_size=128,
                start_training=5
            )
        elif args.delay_type == 'random':
            Model = rlrd.sac_models_rd.Mlp
            agent = SACAgentRD.from_namespace(
                args=args,
                env=env,
                Model=Model,
                memory_size=3000,
                layer_size=256,
                batch_size=128,
                start_training=5,
                OutputNorm=partial(PopArt, beta=0., zero_debias=False)
            )
    elif args.agent_type == 'dcac':
        Model = rlrd.dcac_models.Mlp
        agent = DCACAgent.from_namespace(
            args=args,
            env=env,
            Model=Model,
            input_dims=env.observation_space.shape,
            n_actions=env.action_space.shape[-1],
            max_mem_size=3000,
            layer1_size=256,
            layer2_size=256,
            batch_size=20
        )
    else:
        raise ValueError(f"Unsupported agent type: {args.agent_type}")
    
    """Figure names:"""
    print(f'Training for: {args.agent_name}')

    figure_folder = os.path.join(agent_results_folder, args.agent_name)
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
        
    filename = 'rewards_' + args.agent_name
    avg_filename = '100_steps_average_' + args.agent_name
    
    """Initialize training:"""
    # best_score = env.reward_range[0]
    best_score = None
    score_history = []
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
    
    early_stopping = EarlyStopping(patience=args.patience)
    done_iterations = args.iterations
    print('Training...')
    state = None
    # iterations is the same as epochs
    for i in range(args.iterations):
        steps = 0
        start_time = time.time()
        observation = env.reset()
        done = False
        score = 0
                
        if args.actor_net.lower() == 'lstm':
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, trunc, info = env.step(action)
                steps += 1
                agent.remember(observation, action, reward, observation_, done)
                if not load_checkpoint:
                    agent.learn()
                score += reward
                observation = observation_
        elif args.agent_type == 'sac_mlp_2' and args.delay_type == 'random':
            DelayedSacTraining = partial(
                Training,
                agent=agent,
                Env=partial(
                    rlrd.envs.RandomDelayEnv,
                    id="Pendulum-v0",
                    min_observation_delay=0,
                    sup_observation_delay=1,
                    min_action_delay=0,
                    sup_action_delay=1,
                ),
            )
            
            run(DelayedSacTraining)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        def eval_agent():
            observation, _ = env.reset(flag='eval', eval_len=10)
            done = False
            eval_score = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, trunc, info = env.step(action)
                eval_score += reward
                observation = observation_
        
        if best_score == None or avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                pass
                # TODO: for saving models
                # agent.save_models()
            #eval_agent()
        
        end_time = time.time()
        print(args.env_id, f'Iteration {i+1}|{args.iterations}:', 'Score %.1f' % score,
              'Last 100 iterations avg %.1f' % avg_score,
              'Ep Len %d' % steps, f'Time {int(end_time-start_time)} s')
        
        if i != 0:
            x = [j+1 for j in range(i+1)]
            plot_learning_curve(x, score_history, figure_folder, filename)
        
        early_stopping(score)
        done_iterations = i
        if early_stopping.early_stop:
            break

    if not load_checkpoint:
        with open(os.path.join(figure_folder, 'score_history.pkl'), 'wb') as file:
            pickle.dump(score_history, file)
            
        x = [i+1 for i in range(done_iterations+1)]
        plot_learning_curve(x, score_history, figure_folder, filename)
        plot_avg_learning_curve(x, score_history, figure_folder, avg_filename)


if __name__ == '__main__':
    # Test the random delay agent
    global z_obs
    z_obs = []
    global z_obs_env
    z_obs_env = []
    global z_actions
    z_actions = []
    global z_obs_delay
    z_obs_delay = []
    global z_act_delay
    z_act_delay = []
    global z_rewards
    z_rewards = []
    main()
