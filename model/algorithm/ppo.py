import torch
import torch.nn as nn
import numpy as np
from model.models.one_gaussian import ActorCritic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, batch_size, mini_batch, lam):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.mini_batch = mini_batch
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas) #, weight_decay=1e-5
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory, measurements_summary):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory, measurements_summary).cpu().data.numpy().flatten()

    def select_deterministic_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.deter_act(state).cpu().data.numpy().flatten()

    def select_stochastic_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.stochastic_act(state).cpu().data.numpy().flatten()

    def update(self, memory, last_value):

        # GAE estimate of returns

        memory.values.append(last_value)
        gae = 0
        returns = []
        for step in reversed(range(len(memory.rewards))):
            delta = memory.rewards[step] + self.gamma * memory.values[step + 1] * memory.masks[step] - memory.values[step]
            gae = delta + self.gamma * self.lam * memory.masks[step] * gae
            # prepend to get correct order back
            returns.insert(0, gae + memory.values[step])

        memory.values.pop()


        # Normalizing the rewards:
        returns = torch.tensor(returns).to(device) #ran

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()

        values = torch.squeeze(torch.stack(memory.values)).to(device).detach()
        advantages = returns - values  # ran reward
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # ran
        inds = np.arange(self.batch_size)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            np.random.shuffle(inds)
            for start in range(0, self.batch_size, self.mini_batch):
                end = start + self.mini_batch
                mbinds = inds[start:end]
                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[mbinds], old_actions[mbinds])


                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs[mbinds].detach())

                # Finding Surrogate Loss:
                surr1 = ratios * advantages[mbinds]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[mbinds]
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns[mbinds]) - 0.005 * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())