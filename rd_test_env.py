from wrappers_rd import RandomDelayWrapper
import gym
import numpy as np

env = gym.make('Pendulum-v1', g=9.81)
delayed_env = RandomDelayWrapper(env=env, initial_action=[0.1], act_delay_range=range(0, 5))
delayed_env.reset()
action = 0.5
list_past_actions = []
list_past_observations = []
list_arrival_times_actions = []
list_arrival_times_observations = []
list_cum_rew_brain = []
list_next_action = []
list_rewards = []
for i in range(10):
    obs, reward, done, info = delayed_env.step([action])
    list_past_actions.append(list(delayed_env.past_actions))
    list_past_observations.append(list(delayed_env.past_observations))
    list_arrival_times_actions.append(list(delayed_env.arrival_times_actions))
    list_arrival_times_observations.append(list(delayed_env.arrival_times_observations))
    list_cum_rew_brain.append(delayed_env.cum_rew_brain)
    list_next_action.append(delayed_env.next_action)
    list_rewards.append(reward)
    action = action + 0.1
