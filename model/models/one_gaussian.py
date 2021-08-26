import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from CustomeActivationFunctions import HalfSizeSigmoid, QuarterSizeSigmoid, SoftQuarterSizeSigmoid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCriticOneGaussian(nn.Module):
    def __init__(self, state_dim, action_dim, n_var):
        super(ActorCriticOneGaussian, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.ReLU(),
            nn.Linear(n_var, n_var),
            nn.ReLU(),
            nn.Linear(n_var, n_var),
            nn.ReLU(),
            nn.Linear(n_var, 2*action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.ReLU(),
            nn.Linear(n_var, n_var),
            nn.ReLU(),
            nn.Linear(n_var, n_var),
            nn.ReLU(),
            nn.Linear(n_var, 1)
        )

    def forward(self,x):
        params = self.actor(x)
        try:
            means, sds = torch.split(params, params.shape[1] // 2, dim=1)
        except:
            means, sds = torch.split(params, params.shape[1] // 2, dim=1)

        mean = nn.Softsign()(means)

        sd = QuarterSizeSigmoid()(sds)
        result = MultivariateNormal(mean, torch.diag_embed(sd).to(device))
        return result

    def forward_for_eval(self,x):
        params = self.actor(x)

        mb_tensor_means = torch.Tensor().to(device)
        mb_tensor_sd = torch.Tensor().to(device)
        for i in range(params.shape[0]):

            means, sds = torch.split(params[i], params[i].shape[1] // 2, dim=1)
            mean = nn.Softsign()(means)
            mb_tensor_means = torch.cat([mb_tensor_means, mean], 0)

            sd = QuarterSizeSigmoid()(sds)
            mb_tensor_sd = torch.cat([mb_tensor_sd, sd], 0)

        return MultivariateNormal(mb_tensor_means, torch.diag_embed(mb_tensor_sd).to(device))

    def act(self, state, memory, measurements_summary):
        value = self.critic(state) # ran
        dist = self.forward(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        memory.values.append(value) #ran

        measurements_summary.steering_var.append(dist.variance.cpu().detach().numpy()[0][0])
        measurements_summary.throttle_var.append(dist.variance.cpu().detach().numpy()[0][1])

        return action.detach()

    def deter_act(self, state):
        params = self.actor(state)
        means, sds = torch.split(params, params.shape[1] // 2, dim=1)
        action_mean = nn.Softsign()(means)

        return action_mean.detach()

    def stochastic_act(self, state):
        dist = self.forward(state)
        action = dist.sample()

        return action.detach()

    def evaluate(self, state, action):
        dist = self.forward_for_eval(state)

        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy