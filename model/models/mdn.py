import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, OneHotCategorical
from CustomeActivationFunctions import HalfSizeSigmoid, QuarterSizeSigmoid, SoftQuarterSizeSigmoid
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_var, n_gaussian, action_std=0.0):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.ReLU(),
            nn.Linear(n_var, n_var),
            nn.ReLU(),
            nn.Linear(n_var, n_var),
            nn.ReLU(),
            nn.Linear(n_var, 2 * action_dim * n_gaussian)
        )

        self.alpha = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.ReLU(),
            nn.Linear(n_var, n_var),
            nn.ReLU(),
            nn.Linear(n_var, n_var),
            nn.ReLU(),
            nn.Linear(n_var, n_gaussian),
            nn.Softmax()
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
        self.n_gaussian = n_gaussian


    def forward(self,x):
        params = self.actor(x)
        means, sds = torch.split(params, params.shape[1]//2, dim=1)

        mean = nn.Softsign()(means)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_gaussian, 1))

        sd = QuarterSizeSigmoid()(sds)
        sd = torch.stack(sd.split(sd.shape[1] // self.n_gaussian, 1))

        alpha_params = self.alpha(x)
        alphas = OneHotCategorical(logits=alpha_params)

        return alphas, MultivariateNormal(mean.transpose(0, 1), torch.diag_embed(sd.transpose(0, 1)).to(device))

    def forward_for_eval(self, x):
        params = self.actor(x)
        mb_tensor_means = torch.Tensor().to(device)
        mb_tensor_sd = torch.Tensor().to(device)

        for i in range(params.shape[0]):
            means, sds = torch.split(params[i], params[i].shape[1] // 2, dim=1)
            mean = nn.Softsign()(means)
            mean = torch.stack(mean.split(mean.shape[1] // self.n_gaussian, 1))
            mb_tensor_means = torch.cat([mb_tensor_means,mean],1)

            sd = QuarterSizeSigmoid()(sds)
            sd = torch.stack(sd.split(sd.shape[1] // self.n_gaussian, 1))
            mb_tensor_sd = torch.cat([mb_tensor_sd,sd],1)

        alpha_params = self.alpha(x)
        alphas = OneHotCategorical(logits=alpha_params)

        return alphas, MultivariateNormal(mb_tensor_means.transpose(0, 1), torch.diag_embed(mb_tensor_sd.transpose(0, 1)).to(device))


    def act(self, state, memory, measurements_summary):
        value = self.critic(state)
        alphas, normal_dists = self.forward(state)

        t_alphas = alphas.sample().unsqueeze(2)
        _,max_ind = torch.max(t_alphas, dim=1)
        max_ind=max_ind.cpu().detach().numpy()[0][0]
        chosen_dist_mu = normal_dists.mean[:,max_ind]
        chosen_dist_cov = normal_dists.covariance_matrix[:,max_ind]
        dist = MultivariateNormal(chosen_dist_mu, chosen_dist_cov)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        memory.values.append(value) #ran

        measurements_summary.steering_var.append(dist.variance.cpu().detach().numpy()[0][0])
        measurements_summary.throttle_var.append(dist.variance.cpu().detach().numpy()[0][1])
        measurements_summary.alpha_1.append(alphas.probs.cpu().detach().numpy()[0][0])
        measurements_summary.alpha_2.append(alphas.probs.cpu().detach().numpy()[0][1])
        measurements_summary.alpha_3.append(alphas.probs.cpu().detach().numpy()[0][2])

        return action.detach()

    def deter_act(self, state):
        action_mean = self.actor(state)

        return action_mean.detach()

    def stochastic_act(self, state):

        alphas, normal_dists = self.forward(state)
        t_alphas = alphas.sample().unsqueeze(2)
        _, max_ind = torch.max(t_alphas, dim=1)
        max_ind = max_ind.cpu().detach().numpy()[0][0]
        chosen_dist_mu = normal_dists.mean[:, max_ind]
        chosen_dist_cov = normal_dists.covariance_matrix[:, max_ind]
        dist = MultivariateNormal(chosen_dist_mu, chosen_dist_cov)

        action = dist.sample()

        return action.detach()

    def evaluate(self, state, action):

        alphas, normal_dists = self.forward_for_eval(state)
        t_alphas_a = alphas.sample().transpose(2,1)
        _, max_ind = torch.max(t_alphas_a, dim=1)
        max_ind = max_ind.cpu().detach().numpy()[:,0]
        batch_element_ind = np.arange(0, len(max_ind))
        chosen_dist_mu = normal_dists.mean[batch_element_ind, max_ind,:]
        chosen_dist_cov = normal_dists.covariance_matrix[batch_element_ind, max_ind,:,:]
        dist = MultivariateNormal(chosen_dist_mu, chosen_dist_cov)

        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy