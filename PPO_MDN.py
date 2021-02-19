import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, OneHotCategorical
from CustomeActivationFunctions import HalfSizeSigmoid, QuarterSizeSigmoid, SoftQuarterSizeSigmoid
import numpy as np


import simulator_unity
import tensorflow as tf
import copy
from tensorboardX import SummaryWriter
writer = SummaryWriter()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.masks = []
        self.values = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.masks[:]
        del self.values[:]

class MeasurementsSummary:
    def __init__(self):
        self.steering_var = []
        self.throttle_var = []
        self.alpha_1 = []
        self.alpha_2 = []
        self.alpha_3 = []


    def clear_summary(self):
        del self.steering_var[:]
        del self.throttle_var[:]
        del self.alpha_1[:]
        del self.alpha_2[:]
        del self.alpha_3[:]



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


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, n_gaussian, lr, betas, gamma, K_epochs, eps_clip, batch_size, mini_batch, lam):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.mini_batch = mini_batch
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, n_gaussian).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas) #, weight_decay=1e-5
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, n_gaussian).to(device)

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
        returns = torch.tensor(returns).to(device)

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



def tuple2array(ob):
    converted_obs = np.hstack((ob.sign_theta, ob.trackPos, ob.leftLane, ob.rightLane, ob.speed, ob.current_steering, ob.current_torque,
                               ob.sensors_segments, ob.left_obstacles_front,ob.left_obstacles_back, ob.right_obstacles_front, ob.right_obstacles_back, #ob.curvatures,
                               ob.pre_sign_theta, ob.pre_trackPos, ob.pre_leftLane, ob.pre_rightLane, ob.pre_speed, ob.pre_steering, ob.pre_torque,
                               ob.pre_left_obstacles_front,ob.pre_left_obstacles_back, ob.pre_right_obstacles_front, ob.pre_right_obstacles_back, #ob.pre_curvatures
                               ob.t_2_sign_theta, ob.t_2_trackPos, ob.t_2_leftLane, ob.t_2_rightLane, ob.t_2_speed, ob.t_2_steering, ob.t_2_torque,
                               ob.t_2_left_obstacles_front, ob.t_2_left_obstacles_back,ob.t_2_right_obstacles_front, ob.t_2_left_obstacles_back #ob.t_2_boundaries_sensors# ,
                                  ))
    return converted_obs





def main():

    train = True # True for train False for test
    e_measure_loaded = False
    solved_reward = 9999999999999  # stop training if avg_reward > solved_reward
    log_interval = 1  # print avg reward in the interval
    max_episodes = 50000  # max training episodes
    if train:
        max_timesteps = 4608  # max timesteps in one episode
    else:
        max_timesteps = 99999999
    n_latent_var = 600  # number of units - hidden layer
    update_timestep = batch_size = 1024  # update policy every n timesteps
    mini_batch = 1024
    n_gaussian = 3
    lr = 0.0001
    betas = (0.9, 0.999)
    gamma = 0.95  # discount factor
    lam = 0.95
    K_epochs = 5  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None

    #############################################

    # creating environment
    env = simulator_unity
    windows = 3
    state_dim = 11 * windows + 288
    action_dim = 2

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    measurements_summary = MeasurementsSummary()
    ppo = PPO(state_dim, action_dim, n_latent_var, n_gaussian, lr, betas, gamma, K_epochs, eps_clip, batch_size, mini_batch, lam)
    print(lr, betas)

    print("now we load the weights")
    try:
        ppo.policy.actor.load_state_dict(torch.load("models/best/actor.pt"))
        ppo.policy.critic.load_state_dict(torch.load("models/best/critic.pt"))
        ppo.policy.alpha.load_state_dict(torch.load("models/best/alpha.pt"))

        ppo.policy_old.actor.load_state_dict(torch.load("models/best/actor_old.pt"))
        ppo.policy_old.critic.load_state_dict(torch.load("models/best/critic_old.pt"))
        ppo.policy_old.alpha.load_state_dict(torch.load("models/best/alpha_old.pt"))


    except:
        print("there is no existing models")
    # logging variables
    running_reward = 0

    avg_length = 0
    time_step = 0
    max_length = 0
    stop_flag = 1
    finished_round = False
    rounds_for_update = 2
    time_steps_last_round_updated =0
    rounds_counter = 0
    best_score = -np.inf
    current_cycle_reward = 0
    cycles_avg_reward = []
    measurements = []
    cycle_episodes = 1

    # training loop
    for i_episode in range(1, max_episodes + 1):
        episode_reward = 0
        obs = simulator_unity.get_init_obs()
        initial_speed = float(obs['speed'])

        current_steering = 0.0
        current_torque = 0.0

        pre_obs = simulator_unity.make_pre_observation(obs, current_steering, current_torque)
        t_2_obs = copy.copy(pre_obs)
        human_heading = 0.0
        human_trackPos = 3.0
        ob = simulator_unity.make_observation(obs, current_steering, current_torque, pre_obs, t_2_obs, human_heading, human_trackPos)
        state = tuple2array(ob)

        # choose a sample from GPs
        while initial_speed <np.random.uniform(30,90):
            finished_round = False
            a_drive = [0, 1, 0]
            ob, reward, done,init_pointer, round_number, _ = env.step(a_drive, episode_step=0, pre_obs=pre_obs, t_2_obs=t_2_obs, episode_num=i_episode-1)
            initial_speed = ob.speed *100
            print("initial_speed: ", initial_speed)

            pre_obs = pre_obs._replace(
                current_steering=ob.current_steering,
                current_torque=ob.current_torque,
                speed=ob.speed,
                sign_theta=ob.sign_theta,
                trackPos=ob.trackPos,
                rightLane=ob.rightLane,
                leftLane=ob.leftLane,
                right_obstacles_front=ob.right_obstacles_front,
                right_obstacles_back=ob.right_obstacles_back,
                left_obstacles_front=ob.left_obstacles_front,
                left_obstacles_back=ob.left_obstacles_back,
                sensors_segments=ob.sensors_segments,

            )

            t_2_obs = t_2_obs._replace(
                current_steering=ob.pre_steering,
                current_torque=ob.pre_torque,
                speed=ob.pre_speed,
                sign_theta=ob.pre_sign_theta,
                trackPos=ob.pre_trackPos,
                rightLane=ob.pre_rightLane,
                leftLane=ob.pre_leftLane,
                right_obstacles_front=ob.pre_right_obstacles_front,
                right_obstacles_back=ob.pre_right_obstacles_back,
                left_obstacles_front=ob.pre_left_obstacles_front,
                left_obstacles_back=ob.pre_left_obstacles_back,
                sensors_segments=ob.pre_sensors_segments,

            )

        for t in range(max_timesteps):
            time_steps_last_round_updated+=1
            time_step += 1
            # Running policy_old:
            if train == False:
                action = ppo.select_stochastic_action(state)
            else:
                action = ppo.select_action(state, memory, measurements_summary)
            action = np.clip(action, -1,1)
            ob, reward, done, current_pointer, round_number, _ = env.step(action, episode_step=t, pre_obs=pre_obs, t_2_obs=t_2_obs, episode_num=i_episode-1)
            print("reward: ", reward)
            state = tuple2array(ob)
            print("init pointer: ", init_pointer, "current_pointer: ", current_pointer)
            # Saving reward:
            if train:
                memory.rewards.append(reward)
                memory.masks.append(1-done)

            pre_obs = pre_obs._replace(
                current_steering=ob.current_steering,
                current_torque=ob.current_torque,
                speed=ob.speed,
                sign_theta=ob.sign_theta,
                trackPos=ob.trackPos,
                rightLane=ob.rightLane,
                leftLane=ob.leftLane,
                right_obstacles_front=ob.right_obstacles_front,
                right_obstacles_back=ob.right_obstacles_back,
                left_obstacles_front=ob.left_obstacles_front,
                left_obstacles_back=ob.left_obstacles_back,
                sensors_segments=ob.sensors_segments,

            )

            t_2_obs = t_2_obs._replace(
                current_steering=ob.pre_steering,
                current_torque=ob.pre_torque,
                speed=ob.pre_speed,
                sign_theta=ob.pre_sign_theta,
                trackPos=ob.pre_trackPos,
                rightLane=ob.pre_rightLane,
                leftLane=ob.pre_leftLane,
                right_obstacles_front=ob.pre_right_obstacles_front,
                right_obstacles_back=ob.pre_right_obstacles_back,
                left_obstacles_front=ob.pre_left_obstacles_front,
                left_obstacles_back=ob.pre_left_obstacles_back,
                sensors_segments=ob.pre_sensors_segments,

            )

            episode_reward +=reward
            current_cycle_reward += reward

            print("time_step: ", time_step)

            # check if agent finish a full round
            if (current_pointer>=init_pointer and current_pointer<=init_pointer+2) and t>100:
                if time_steps_last_round_updated>100:
                    rounds_counter +=1
                    time_steps_last_round_updated=0

                print("rounds_counter: ", rounds_counter)
                if (rounds_counter)%rounds_for_update ==0:
                    rounds_counter=0
                    # time_steps_last_round_updated=0
                    finished_round = True
                    print("finished a full round - update parameters!")

            if train:
            # update if its time
                if time_step % update_timestep == 0 or finished_round:# or time_step == 2944:

                    env.end()
                    state_eval = torch.FloatTensor(state.reshape(1, -1)).to(device)
                    ppo.batch_size = time_step
                    last_value = ppo.policy_old.critic(state_eval)
                    ppo.update(memory, last_value)
                    memory.clear_memory()

                    time_step = 0

                    #arguments for the plots
                    cycles_avg_reward = current_cycle_reward/cycle_episodes
                    measurements.append(cycles_avg_reward)
                    current_cycle_reward = 0
                    cycle_episodes = 0


                    break
                running_reward += reward

            if done:
                env.end()
                break


        avg_length += t

        if train:
            if avg_length > max_length:
                max_length = avg_length
                substraction = (max_length*2)%mini_batch
                update_timestep = max(512, max_length*2 - substraction)
            print("update_timestep: ", update_timestep)


        writer.add_scalar('data/total_length', avg_length, i_episode)
        writer.add_scalar('data/total_reward', running_reward, i_episode)
        writer.add_scalar('var/steering_var', np.mean(measurements_summary.steering_var), i_episode)
        writer.add_scalar('var/throttle_var', np.mean(measurements_summary.throttle_var), i_episode)
        writer.add_scalar('var/max_steering_var', np.max(measurements_summary.steering_var), i_episode)
        writer.add_scalar('var/max_throttle_var', np.max(measurements_summary.throttle_var), i_episode)
        writer.add_scalar('alphas/alpha_1', np.mean(measurements_summary.alpha_1), i_episode)
        writer.add_scalar('alphas/alpha_2', np.mean(measurements_summary.alpha_2), i_episode)
        writer.add_scalar('alphas/alpha_3', np.mean(measurements_summary.alpha_3), i_episode)
        writer.add_histogram('hist/alpha_1', measurements_summary.alpha_1, i_episode)
        writer.add_histogram('hist/alpha_2', measurements_summary.alpha_2, i_episode)
        writer.add_histogram('hist/alpha_3', measurements_summary.alpha_3, i_episode)

        measurements_summary.clear_summary()

        print("episode: ", i_episode)
        cycle_episodes +=1
        print("cycle_episodes: ", cycle_episodes)


        # # stop training if avg_reward > solved_reward
        if running_reward > solved_reward:
            print("########## Solved! ##########")
            break

        # logging
        if i_episode % log_interval == 0:


            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

        if train:
            if episode_reward > best_score:
                best_score = episode_reward
                # save the best model:
                torch.save(ppo.policy.actor.state_dict(), "models/best/actor.pt")
                torch.save(ppo.policy.critic.state_dict(), "models/best/critic.pt")
                torch.save(ppo.policy.alpha.state_dict(), "models/best/alpha.pt")

                torch.save(ppo.policy_old.actor.state_dict(), "models/best/actor_old.pt")
                torch.save(ppo.policy_old.critic.state_dict(), "models/best/critic_old.pt")
                torch.save(ppo.policy_old.alpha.state_dict(), "models/best/alpha_old.pt")

            #save the last model:
            torch.save(ppo.policy.actor.state_dict(), "models/actor.pt")
            torch.save(ppo.policy.critic.state_dict(), "models/critic.pt")
            torch.save(ppo.policy.alpha.state_dict(), "models/alpha.pt")

            torch.save(ppo.policy_old.actor.state_dict(), "models/actor_old.pt")
            torch.save(ppo.policy_old.critic.state_dict(), "models/critic_old.pt")
            torch.save(ppo.policy_old.alpha.state_dict(), "models/alpha_old.pt")



if __name__ == '__main__':
    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K

    K.set_session(sess)
    main()

