"""
Created on Monday Dec 01 2023
@author: Esmaeel Mohammadi

# =============================================================================
# This script is used to train and test Soft Actor-Critic algorithm:
    1- Create an instance of environment
    2- Train SAC on the environment
    3- Test the trained policy
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

import torch

from PhosphorusEnvironment_delay import PhosphorusEnvironment as Env_delay

from sac_torch import Agent
from sac_utils import plot_learning_curve, plot_avg_learning_curve, EarlyStopping

import time

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# %%
if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count()

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Config the agent training args
    parser = argparse.ArgumentParser(
        description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--setting', type=str, required=False,
                        default='', help='the model setting')
    parser.add_argument('--experiment', type=int, required=False,
                        default=1, help='the experiment: 1, 2, 3, 4')
    parser.add_argument('--const_el', type=int, required=False,
                        default=360, help='the constant episode length')
    parser.add_argument('--min_el', type=int, required=False,
                        default=90, help='the min episode length')
    parser.add_argument('--max_el', type=int, required=False,
                        default=1440, help='the max episode length')
    parser.add_argument('--delay', type=int, required=False,
                        default=90, help='the max delay of the system')
    parser.add_argument('--retrain', type=bool, required=False,
                        default=True, help='either to use the retrained or not')
    parser.add_argument('--retr_chkpt', type=str, required=False,
                        default='', help='the retrain checkpoint name')
    parser.add_argument('--env_id', type=str, required=False,
                        default='P_const_delay', help='the id of the environment')
    parser.add_argument('--agent_name', type=str, required=False,
                        default='', help='the name of the agent')
    parser.add_argument('--early_stopping', type=bool, required=False,
                        default=True, help='perform early stopping or not')
    parser.add_argument('--patience', type=int, required=False,
                        default=100, help='the patience for early stopping')

    args = parser.parse_args()

    args.setting = 'LSTM_IOPTQCfP_13F_1Out_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256H_2L'
    args.experiment = 1
    args.const_el = 10
    args.min_el = 90
    args.max_el = 1440
    args.const_delay = False
    args.delay = 1
    args.retrain = True
    args.retr_chkpt = 'E3N1 - 50Ep_ActA_SGap_240EL_Batch_20SB_dilate_all_0.6A_1e-06LR_Vdtw.pth'
    args.env_id = 'P_const_delay'
    args.iterations = 1
    args.init_policy = 'LSTM_IOPTQCfP_13F_3Out_timeF_Unscaled_90Seq_16Batch_1e-06LR_80H_1L'
    args.early_stopping = True
    args.patience = int(np.ceil(0.1*args.iterations))

    el = f'{args.max_el}MaxEL' if args.experiment in [
        2, 4] else f'{args.const_el}EL'

    agent_name = '{}_E{}_{}_{}Itr'.format(
        args.env_id, args.experiment, el, args.iterations)

    args.agent_name = agent_name
    
    print(f'Training for: {agent_name}')

    FIGURES_PATH = './agent_results/'
    figure_folder = FIGURES_PATH + agent_name + '/'
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)

    num_envs = 1
    env = Env_delay(args.setting, num_envs, device, experiment=args.experiment, const_el=args.const_el,
                    min_el=args.min_el, max_el=args.max_el, const_delay=args.const_delay, delay=args.delay, retrain=args.retrain,
                    retr_chkpt=args.retr_chkpt, fig_folder=figure_folder, mode='not_live')

    agent = Agent(alpha=0.0003, beta=0.0003, max_size=3000, reward_scale=2, env_id=args.env_id,
                  input_dims=env.observation_space.shape, tau=0.005,
                  env=env, batch_size=32, layer1_size=256, layer2_size=256,
                  n_actions=env.action_space.shape[0], init_policy=args.init_policy,
                  actor_net='LSTM', device=device, agent_args=args)

    stop
    filename = 'rewards_' + agent_name
    avg_filename = '100_steps_average_' + agent_name

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
