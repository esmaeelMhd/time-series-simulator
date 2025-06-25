import torch
import torch.nn.functional as F
from functools import reduce

class CombinedSAC:
    def __init__(self, memory, actor, critic_1, critic_2, target_critic_1, target_critic_2, value, target_value, actor_optimizer, critic_optimizer, value_optimizer, outputnorm, outputnorm_target, batch_size, discount, tau, reward_scale, entropy_scale, device):
        self.memory = memory
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = target_critic_1
        self.target_critic_2 = target_critic_2
        self.value = value
        self.target_value = target_value
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.value_optimizer = value_optimizer
        self.outputnorm = outputnorm
        self.outputnorm_target = outputnorm_target
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.reward_scale = reward_scale
        self.entropy_scale = entropy_scale
        self.device = device

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done).to(self.device)

        # Sample actions using reparameterization trick
        new_actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)

        # Compute critic values using both critics for current and next state
        with torch.no_grad():
            next_actions, _ = self.actor.sample_normal(new_state, reparameterize=False)
            next_value_1 = self.target_critic_1(new_state, next_actions)
            next_value_2 = self.target_critic_2(new_state, next_actions)
            next_value = torch.min(next_value_1, next_value_2).view(-1)
            next_value[done] = 0.0
            next_value = self.outputnorm_target.unnormalize(next_value)

        # Entropy bonus
        next_action_entropy = -(1. - done) * self.discount * log_probs
        reward_components = torch.cat((self.reward_scale * reward[:, None], self.entropy_scale * next_action_entropy[:, None]), dim=1)
        value_target = reward_components + (1. - done[:, None]) * self.discount * next_value
        normalized_value_target = self.outputnorm.update(value_target)

        # Compute losses for both critics and update
        self.critic_optimizer.zero_grad()
        q1_old_policy = self.critic_1(state, action).view(-1)
        q2_old_policy = self.critic_2(state, action).view(-1)
        critic_loss = 0.5 * (F.mse_loss(q1_old_policy, normalized_value_target) + F.mse_loss(q2_old_policy, normalized_value_target))
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        q1_new_policy = self.critic_1(state, new_actions).view(-1)
        q2_new_policy = self.critic_2(state, new_actions).view(-1)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        actor_loss = (log_probs - critic_value).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_network_parameters()

    def update_network_parameters(self):
        # Helper function to soft update parameters
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)
        soft_update(self.target_value, self.value, self.tau)
        # Also update the normalization parameters
        exponential_moving_average(self.outputnorm_target.parameters(), self.outputnorm.parameters(), self.tau)
