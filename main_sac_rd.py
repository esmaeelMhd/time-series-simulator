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
import argparse
import os
import pickle
import random
import time
import logging
# for the cloud
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import torch

from PhosphorusEnvironment_delay import PhosphorusEnvironment as Env_delay
from env_wrapper import RandomDelayWrapper

from sac_torch import Agent
from sac_utils import plot_learning_curve, plot_avg_learning_curve, EarlyStopping

#%%

def make_args():
    parser = argparse.ArgumentParser(description='Configure experiment parameters for delayed RL training')

    # Define arguments with defaults
    parser.add_argument('--setting', type=str, default=None, help='the model setting')
    parser.add_argument('--policy', type=str, default=None, help='the policy method')
    parser.add_argument('--title', type=str, default=None, help='title for training the agent')
    parser.add_argument('--scaled', type=bool, default=None, help='either to scale the obs, rewards, ...')
    parser.add_argument('--experiment', type=int, default=None, help='the experiment: 1, 2, 3, 4')
    parser.add_argument('--const_el', type=int, default=None, help='the constant episode length')
    parser.add_argument('--min_el', type=int, default=None, help='the min episode length')
    parser.add_argument('--max_el', type=int, default=None, help='the max episode length')
    parser.add_argument('--delay_type', type=str, default=None, help='the delay type: constant or random')
    parser.add_argument('--max_delay', type=int, default=None, help='the max delay of the system')
    parser.add_argument('--act_delay_max', type=int, default=None, help='the max delay of the actions')
    parser.add_argument('--obs_delay_max', type=int, default=None, help='the max delay of the observations')
    parser.add_argument('--retrain', type=bool, default=None, help='either to use the retrained or not')
    parser.add_argument('--retr_chkpt', type=str, default=None, help='the retrain checkpoint name')
    parser.add_argument('--env_id', type=str, default=None, help='the id of the environment')
    parser.add_argument('--agent_name', type=str, default=None, help='the name of the agent')
    parser.add_argument('--early_stopping', type=bool, default=None, help='perform early stopping or not')
    parser.add_argument('--patience', type=int, default=None, help='the patience for early stopping')

    return parser.parse_args()

def set_defaults(args):
    # Default values mapping
    defaults = {
        'setting': 'LSTM_IOPTQCfP_13F_1Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L',
        # policy: 1- sac, 2- sac_cont_delay, 3- sac_random_delay, 4- dcac
        'policy': 'sac_const_delay',
        'title': 'Random Delay',
        'scaled': True,
        'experiment': 1,
        'const_el': 10,
        'min_el': 90,
        'max_el': 1440,
        'delay_type': 'random',
        'max_delay': 3,
        'act_delay_max': 10,
        'obs_delay_max': 300,
        'retrain': True,
        'retr_chkpt': 'E3N1 - 50Ep_SGap_240EL_Batch_20SB_d_all_0.6A_1e-06LR_Vdtw.pth',
        'env_id': 'P_const_delay',
        'iterations': 1,
        'init_policy': 'LSTM_IOPTQCfP_13F_3Out_timeF_Unscaled_90Seq_16Batch_1e-06LR_80H_1L',
        'early_stopping': True
    }
    
    # Apply defaults if not provided in command line
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    
    # Dynamic settings
    if args.patience is None:
        args.patience = int(np.ceil(0.1 * args.iterations))
    
    # Calculate `agent_name` based on other parameters
    el = f'{args.max_el}MaxEL' if args.experiment in [2, 4] else f'{args.const_el}EL'
    agent_name = f'{args.env_id}_E{args.experiment}_{el}_{args.iterations}Itr'
    args.agent_name = agent_name

# %%
def main():
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count()

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Config the agent training args
    args = make_args()
    set_defaults(args)
    
    print(f'Training for: {args.agent_name}')

    FIGURES_PATH = './agent_results/'
    figure_folder = FIGURES_PATH + args.agent_name + '/'
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    
    # Environment instance
    # model_args = ModelArgs(**vars(args), checkpoint="")
    num_envs = 1
    env = Env_delay(args, num_envs, device, fig_folder=figure_folder, mode='not_live')
    
    # Env Wrapper if the delay type is random
    if args.delay_type == 'random':
        delayed_env = RandomDelayWrapper(env=env, initial_action=np.array([0.1, 0.1, 0.1]), 
                                         obs_delay_range=range(0, 10), act_delay_range=range(0, 3))
        delayed_env.reset()
        
    def test_env_wrapper(delayed_env):
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
            obs, reward, done, info = delayed_env.step(action)
            list_past_actions.append(list(delayed_env.past_actions))
            list_past_observations.append(list(delayed_env.past_observations))
            list_arrival_times_actions.append(list(delayed_env.arrival_times_actions))
            list_arrival_times_observations.append(list(delayed_env.arrival_times_observations))
            list_cum_rew_brain.append(delayed_env.cum_rew_brain)
            list_next_action.append(delayed_env.next_action)
            list_rewards.append(reward)
            list_obs.append(obs)
            action = action + 0.1
    
    test_env_wrapper(delayed_env)  

    # Agent
    agent = Agent(alpha=0.0003, beta=0.0003, max_size=3000, reward_scale=2, env_id=args.env_id,
                  input_dims=env.observation_space.shape, tau=0.005,
                  env=env, batch_size=32, layer1_size=256, layer2_size=256,
                  n_actions=env.action_space.shape[0], init_policy=args.init_policy,
                  actor_net='LSTM', device=device, agent_args=args)

    filename = 'rewards_' + args.agent_name
    avg_filename = '100_steps_average_' + args.agent_name

    # best_score = env.reward_range[0]
    best_score = None
    score_history = []
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    z_obs = env.observations
    z_target = env.targets
    z_actions = env.actions
    z_sequence = env.sequences
    z_rewards = env.rewards
    
    early_stopping = EarlyStopping(patience=args.patience)
    done_iterations = args.iterations
    print('Training...')
    # iterations is the same as epochs
    for i in range(args.iterations):
        steps = 0
        start_time = time.time()
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            steps += 1
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            score += reward
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        def eval_agent():
            observation = env.reset(flag='eval', eval_len=10)
            done = False
            eval_score = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                eval_score += reward
                observation = observation_
        
        if best_score == None or avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
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
        with open(figure_folder + 'score_history' + '.pkl', 'wb') as file:
            pickle.dump(score_history, file)
            
        x = [i+1 for i in range(done_iterations+1)]
        plot_learning_curve(x, score_history, figure_folder, filename)
        plot_avg_learning_curve(x, score_history, figure_folder, avg_filename)


if __name__ == '__main__':
    main()