import torch
import numpy as np
import simulator_unity
import tensorflow as tf
from keras import backend as K
import copy
from tensorboardX import SummaryWriter
from model.memory.memory import Memory, MeasurementsSummary
from model.algorithm.ppo import PPO
from utils.utils import tuple2array, get_t1_obs, get_t2_obs, save_model

writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SENSOR_SEGMENT_DIM = 288
NON_SENSOR_SEGMENT_DIM = 11
OBSERVATION_TIME_WINDOW = 3
ACTION_DIM = 2


def main():
    ############## Hyperparameters ##############

    train = True # True for train False for test
    max_episodes = 50000  # max training episodes
    if train:
        max_timesteps = 4608  # max timesteps in one episode
    else:
        max_timesteps = 99999999
    n_latent_var = 600  # number of variables in hidden layer
    update_timestep = batch_size = 512  # update policy every n timesteps
    mini_batch = 1024


    lr = 0.0001
    betas = (0.9, 0.999)
    gamma = 0.96  # discount factor
    lam = 0.95
    K_epochs = 5  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    # epsilon = 1
    # epsilon_decay = 0.9999
    #############################################

    # creating environment
    env = simulator_unity

    state_dim = NON_SENSOR_SEGMENT_DIM * OBSERVATION_TIME_WINDOW + SENSOR_SEGMENT_DIM  # + 19 #- 2
    action_dim = ACTION_DIM

    memory = Memory()
    measurements_summary = MeasurementsSummary()

    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, batch_size, mini_batch, lam)
    print(lr, betas)

    print("now we load the weights")
    try:
        ppo.policy.actor.load_state_dict(torch.load("models/best/actor.pt"))
        ppo.policy.critic.load_state_dict(torch.load("models/best/critic.pt"))

        ppo.policy_old.actor.load_state_dict(torch.load("models/best/actor_old.pt"))
        ppo.policy_old.critic.load_state_dict(torch.load("models/best/critic_old.pt"))

    except:
        print("there is no existing models")
    # logging variables
    running_reward = 0
    # std = ppo.policy.action_var[0]
    # ppo.policy.action_var[0] += 1e-2
    episode_length = 0
    time_step = 0
    tot_time_step = 0
    # max_length = 5888
    # max_length = 2944
    # max_length = 2048
    max_length = 512

    # max_length = 0
    #avg_throttle = 0
    stop_flag = 1
    finished_round = False
    rounds_for_update = 2
    time_steps_last_round_updated = 0
    rounds_counter = 0
    best_score = -np.inf
    current_cycle_reward = 0
    cycles_avg_reward = []
    measurements = []
    total_episodes = 1

    # training loop
    for i_episode in range(1, max_episodes + 1):

        episode_reward = 0
        obs = simulator_unity.get_init_obs()
        initial_speed = float(obs['speed'])

        current_steering = 0.0
        current_torque = 0.0

        t1_obs = simulator_unity.make_pre_observation(obs, current_steering, current_torque)
        t2_obs = copy.copy(t1_obs)
        human_heading = 0.0
        human_trackPos = 3.0
        ob = simulator_unity.make_observation(obs, current_steering, current_torque, t1_obs, t2_obs, human_heading, human_trackPos)
        state = tuple2array(ob)

        while initial_speed < np.random.uniform(30,90):
            finished_round = False
            a_drive = [0, 1, 0]
            ob, reward, done, init_pointer, _, _ = env.step(a_drive, episode_step=0, pre_obs=t1_obs, t_2_obs=t2_obs, episode_num=i_episode-1)
            initial_speed = ob.speed *100
            print("initial_speed: ", initial_speed)

            t1_obs = get_t1_obs(t1_obs, ob)
            t2_obs = get_t2_obs(t2_obs, ob)

        for t in range(max_timesteps):
            time_steps_last_round_updated+=1
            time_step += 1
            # Running policy_old:
            if train == False:
                action = ppo.select_stochastic_action(state)
                # action = ppo.select_deterministic_action(state)
            else:
                action = ppo.select_action(state, memory, measurements_summary)

            action = np.clip(action, -1,1)
            ob, reward, done, current_pointer, _, _ = env.step(action, episode_step=t, pre_obs=t1_obs, t_2_obs=t2_obs, episode_num=i_episode-1)
            print("reward: ", reward)
            state = tuple2array(ob)
            print("init pointer: ", init_pointer, "current_pointer: ", current_pointer)

            t1_obs = get_t1_obs(t1_obs, ob)
            t2_obs = get_t2_obs(t2_obs, ob)

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
                    finished_round = True
                    print("finished a full round - update parameters!")

            if train:
                memory.rewards.append(reward)
                memory.masks.append(1 - done)
                # update if its time
                if time_step % update_timestep == 0 or finished_round:

                    env.end()
                    state_eval = torch.FloatTensor(state.reshape(1, -1)).to(device)
                    ppo.batch_size = time_step
                    last_value = ppo.policy_old.critic(state_eval)
                    ppo.update(memory, last_value)
                    memory.clear_memory()
                    time_step = 0

                    #arguments for the plots
                    cycles_avg_reward = current_cycle_reward/total_episodes
                    measurements.append(cycles_avg_reward)
                    current_cycle_reward = 0
                    total_episodes = 0
                    break

                running_reward += reward

            if done:
                env.end()
                break


        ###################################### after episode termination ######################################

        episode_length += t

        if train:
            if episode_length+1 >= max_length:
                print("episode_length: ", episode_length)
                max_length = episode_length+1
                substraction = (max_length*2)%mini_batch
                print("substraction: ", episode_length)
                tmp = max(512, max_length*2 - substraction)
                update_timestep = min(tmp, max_timesteps) # 5.11.19
            print("update_timestep: ", update_timestep)



        writer.add_scalar('data/total_length', episode_length, i_episode)
        writer.add_scalar('data/total_reward', running_reward, i_episode)
        writer.add_scalar('var/steering_var', np.mean(measurements_summary.steering_var), i_episode)
        writer.add_scalar('var/throttle_var', np.mean(measurements_summary.throttle_var), i_episode)
        writer.add_scalar('var/max_steering_var', np.max(measurements_summary.steering_var), i_episode)
        writer.add_scalar('var/max_throttle_var', np.max(measurements_summary.throttle_var), i_episode)

        measurements_summary.clear_summary()

        total_episodes +=1
        print("total_episodes: ", total_episodes)
        print('Episode {} \t length: {} \t reward: {}'.format(i_episode, episode_length, running_reward))
        running_reward = 0
        episode_length = 0

        if train:
            save_model(ppo, episode_reward, best_score)


if __name__ == '__main__':

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    K.set_session(sess)
    main()

