
import argparse
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
ALGORITHM_TYPE = ['PLAIN', 'MDN', 'IRL']

INIT_CURRENT_STEERING = 0.0
INIT_CURRENT_TORQUE = 0.0
INIT_HUMAN_HEADING = 0.0
INIT_HUMAN_TRACKPOS = 3.0

MIN_INIT_SPEED = 30
MAX_INIT_SPEED = 90


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='PLAIN', required=False, choices=ALGORITHM_TYPE)
    parser.add_argument('--train', default=False, action='store_true', help='true for training, else (test) false')
    parser.add_argument('--human-gp-data', type=str, required=False, help='path to load the modeled human data based on the GPs')
    parser.add_argument('--human-data', type=str, required=False, help='path to load the human data')
    parser.add_argument('--max-episodes', type=int, default=9999999, help='max training episodes')
    parser.add_argument('--max-timesteps', type=int, default=4608, help='max timesteps in one episode')
    parser.add_argument('--units', type=int, default=600, help='number of units in the hidden layer')
    parser.add_argument('--batch-size', type=int, default=512, help='batch size')
    parser.add_argument('--mini-batch-size', type=int, default=256, help='mini batch size')
    parser.add_argument('--max-length', type=int, default=512, help='max length parameter for the dynamic update algorithm')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='betas for Adam optimizer')
    parser.add_argument('--gamma', type=float, default=0.96, help='discount factor')
    parser.add_argument('--lam', type=float, default=0.95, help='lambda for GAE')
    parser.add_argument('--k-epochs', type=int, default=5, help='policy update number of epochs')
    parser.add_argument('--eps-clip', type=float, default=0.2, help='clip parameter for PPO')
    args = parser.parse_args()

    # creating environment
    env = simulator_unity

    state_dim = NON_SENSOR_SEGMENT_DIM * OBSERVATION_TIME_WINDOW + SENSOR_SEGMENT_DIM  # + 19 #- 2
    action_dim = ACTION_DIM

    memory = Memory()
    measurements_summary = MeasurementsSummary()

    ppo = PPO(state_dim, action_dim, args.units, args.lr, args.betas, args.gamma, args.k_epochs, args.eps_clip, args.batch_size, args.mini_batch_size, args.lam)
    # print(lr, betas)

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
    episode_length = 0
    time_step = 0

    finished_round = False
    rounds_for_update = 2
    time_steps_last_round_updated = 0
    rounds_counter = 0
    best_score = -np.inf
    current_cycle_reward = 0
    measurements = []
    total_episodes = 1

    update_timestep = args.batch_size
    if not args.train:
        args.max_timesteps = 99999999

    # training loop
    for i_episode in range(1, args.max_episodes + 1):

        episode_reward = 0
        obs = simulator_unity.get_init_obs()
        initial_speed = float(obs['speed'])

        t1_obs = simulator_unity.make_pre_observation(obs, INIT_CURRENT_STEERING, INIT_CURRENT_TORQUE)
        t2_obs = copy.copy(t1_obs)

        ob = simulator_unity.make_observation(obs, INIT_CURRENT_STEERING, INIT_CURRENT_TORQUE, t1_obs, t2_obs, INIT_HUMAN_HEADING, INIT_HUMAN_TRACKPOS)
        state = tuple2array(ob)

        while initial_speed < np.random.uniform(MIN_INIT_SPEED, MAX_INIT_SPEED):
            finished_round = False
            a_drive = [0, 1, 0]
            ob, reward, done, init_pointer, _, _ = env.step(a_drive, episode_step=0, pre_obs=t1_obs, t_2_obs=t2_obs, episode_num=i_episode-1)
            initial_speed = ob.speed *100
            print("initial_speed: ", initial_speed)

            t1_obs = get_t1_obs(t1_obs, ob)
            t2_obs = get_t2_obs(t2_obs, ob)

        for t in range(args.max_timesteps):
            time_steps_last_round_updated+=1
            time_step += 1
            # Running policy_old:
            if args.train == False:
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

            if args.train:
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

        if args.train:
            if episode_length+1 >= args.max_length:
                print("episode_length: ", episode_length)
                args.max_length = episode_length+1
                substraction = (args.max_length*2)%args.mini_batch
                print("substraction: ", episode_length)
                tmp = max(512, args.max_length*2 - substraction)
                update_timestep = min(tmp, args.max_timesteps) # 5.11.19
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

        if args.train:
            save_model(ppo, episode_reward, best_score)


if __name__ == '__main__':

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    K.set_session(sess)
    main()

