import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
colors = [color for color in mcolors.TABLEAU_COLORS.values()]

plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.xmargin'] = 0.02
plt.rcParams['axes.ymargin'] = 0.04
plt.rcParams['axes.labelsize'] = 10

plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)
fig_dpi=500
        
def plot_learning_curve(x, scores, fig_path, figure_file):
    plt.figure(figsize=(3.2, 3), dpi=fig_dpi)
    plt.plot(x, scores, 'x-', color=colors[0], markersize=0.2)
    plt.grid(visible=True, which='major', color='lightgray', linewidth=0.0025)
    plt.grid(visible=True, which='minor', color='lightgray', linewidth=0.0025)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward of the episode', labelpad=10)
    plt.subplots_adjust(left=0.24, right=0.98, bottom=0.18, top=0.98)
    plt.savefig(os.path.join(fig_path, figure_file+'.pdf'))
    
def plot_avg_learning_curve(x, scores, fig_path, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.figure(figsize=(3.2, 3), dpi=fig_dpi)
    plt.plot(x, running_avg, 'x-', color=colors[1], markersize=0.2)
    plt.grid(visible=True, which='major', color='lightgray', linewidth=0.0025)
    plt.grid(visible=True, which='minor', color='lightgray', linewidth=0.0025)
    plt.xlabel('Episodes')
    plt.ylabel('Average reward of the previous 100 steps', labelpad=10)
    plt.subplots_adjust(left=0.24, right=0.98, bottom=0.18, top=0.98)
    plt.savefig(os.path.join(fig_path, figure_file+'.pdf'))
    
class EarlyStopping:
    def __init__(self, patience=100):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.reward_min = -np.Inf

    def __call__(self, reward):
        score = reward
        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

