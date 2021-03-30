import torch
import torch.nn as nn
import numpy as np
import math
from model.models.gail_models import ActorCritic, Discriminator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GAIL:
    def __init__(self, state_dim, action_dim, args):# n_latent_var, lr, betas, gamma, K_epochs, eps_clip, batch_size, mini_batch, lam, discrim_update_num):
        self.lr = args.lr
        self.units = args.units
        self.betas = args.betas
        self.gamma = args.gamma
        self.lam = args.lam
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.eps_clip = args.eps_clip
        self.k_epochs = args.k_epochs
        self.discrim_update_num = args.discrim_update_num

        self.policy = ActorCritic(state_dim, action_dim, self.units).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas) #, weight_decay=1e-5

        self.discriminator = Discriminator(state_dim + action_dim, self.units).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr*0.1, betas=self.betas) #, weight_decay=1e-5
        self.policy_old = ActorCritic(state_dim, action_dim, self.units).to(device)

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

    def get_reward(self, state, action):
        state = torch.Tensor(state)
        action = torch.Tensor(action)
        state_action = torch.cat([state, action])
        with torch.no_grad():
            return -math.log(self.discriminator(state_action)[0].item())

    def update_actor_critic(self, memory, last_value):

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
        for _ in range(self.k_epochs):
            np.random.shuffle(inds)
            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
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

    def train_discriminator(self, memory, demonstrations):
        # memory = np.array(memory)
        # states = np.vstack(memory[:, 0])
        # actions = list(memory[:, 1])
        #
        # states = torch.Tensor(states)
        # actions = torch.Tensor(actions)

        states = torch.stack(memory.states).squeeze(dim=1).to(device).detach()
        actions = torch.stack(memory.actions).squeeze(dim=1).to(device).detach()

        criterion = torch.nn.BCELoss()

        for _ in range(self.discrim_update_num):
            learner = self.discriminator(torch.cat([actions, states], dim=1))
            demonstrations = torch.Tensor(demonstrations)
            expert = self.discriminator(demonstrations)

            discrim_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
                           criterion(expert, torch.zeros((demonstrations.shape[0], 1)))

            self.discriminator_optimizer.zero_grad()
            discrim_loss.backward()
            self.discriminator_optimizer.step()

        expert_acc = ((self.discriminator(demonstrations) < 0.5).float()).mean()
        learner_acc = ((self.discriminator(torch.cat([states, actions], dim=1)) > 0.5).float()).mean()

        return expert_acc, learner_acc