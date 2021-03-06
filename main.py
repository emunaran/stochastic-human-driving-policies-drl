import argparse
import torch
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
import copy
from tensorboardX import SummaryWriter
from model.memory.memory import Memory, MeasurementsSummary
from model.algorithm.ppo import PPO
from model.algorithm.gail import GAIL
from utils.utils import tuple2array, get_t1_obs, get_t2_obs, save_model
from utils.logger import Logger
import logging
from utils import constants
from dataclasses import asdict
from env.simulator import Simulator, server, env_utils


writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='PLAIN', choices=list(asdict(constants.ALGORITHM_TYPE).values()))
    parser.add_argument('--train', default=False, action='store_true', help='true for training, else (test) false')
    parser.add_argument('--human-gp-data', type=str, default='./data/expert/gp/GP_expert.csv', help='path to load the modeled human data based on the GPs')
    parser.add_argument('--human-demonstrations', type=str, default='./data/expert/demonstrations/demonstrations.csv', help='path to load the human demonstrations')
    parser.add_argument('--max-episodes', type=int, default=9999999, help='max training episodes')
    parser.add_argument('--max-timesteps', type=int, default=4608, help='max timesteps in one episode')
    parser.add_argument('--units', type=int, default=600, help='number of units in the hidden layer')
    parser.add_argument('--batch-size', type=int, default=512, help='batch size')
    parser.add_argument('--mini-batch-size', type=int, default=256, help='mini batch size')
    parser.add_argument('--max-length', type=int, default=512, help='max length parameter for the dynamic update algorithm')
    parser.add_argument('--discrim-update-num', type=int, default=1, help='number of updates for discriminator')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='betas for Adam optimizer')
    parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
    parser.add_argument('--lam', type=float, default=0.95, help='lambda for GAE')
    parser.add_argument('--k-epochs', type=int, default=5, help='policy update number of epochs')
    parser.add_argument('--eps-clip', type=float, default=0.2, help='clip parameter for PPO')
    parser.add_argument('--n-gaussian', type=int, default=3, help='number of gaussian for MDN')
    args = parser.parse_args()

    Logger.init_logger(log_path='log_files', log_name=args.algorithm)
    logging.info(f'logging for {args.algorithm} was started')

    # creating environment
    # env = simulator_unity # this is the old version of env - use it if something breaks.
    env = Simulator(args.train, args.algorithm)

    state_dim = constants.NON_SENSOR_SEGMENT_DIM * constants.OBSERVATION_TIME_WINDOW + constants.SENSOR_SEGMENT_DIM
    action_dim = constants.ACTION_DIM

    memory = Memory()
    measurements_summary = MeasurementsSummary()

    if args.algorithm == constants.ALGORITHM_TYPE.GAIL:
        # load human demonstrations
        human_demonstrations = pd.read_csv(args.human_demonstrations)
        human_demonstrations = np.array(human_demonstrations)
        print("human_demonstrations.shape", human_demonstrations.shape)
        model = GAIL(state_dim, action_dim, args)
    else:
        model = PPO(state_dim, action_dim, args)

    print("now we load the weights")
    try:
        model.policy.actor.load_state_dict(torch.load(f"saved_models/{args.algorithm}/best/actor.pt"))
        model.policy.critic.load_state_dict(torch.load(f"saved_models/{args.algorithm}/best/critic.pt"))

        model.policy_old.actor.load_state_dict(torch.load(f"saved_models/{args.algorithm}/best/actor_old.pt"))
        model.policy_old.critic.load_state_dict(torch.load(f"saved_models/{args.algorithm}/best/critic_old.pt"))

        if args.algorithm == constants.ALGORITHM_TYPE.GAIL:
            model.discriminator.load_state_dict(torch.load(f"saved_models/{args.algorithm}/best/discriminator.pt"))
    except FileNotFoundError:
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
        obs = server.get_init_obs()
        initial_speed = float(obs['speed'])

        t1_obs = env_utils.make_pre_observation(obs, constants.INIT_CURRENT_STEERING, constants.INIT_CURRENT_TORQUE)
        t2_obs = copy.copy(t1_obs)

        ob = env_utils.make_observation(obs, constants.INIT_CURRENT_STEERING, constants.INIT_CURRENT_TORQUE, t1_obs, t2_obs, constants.INIT_HUMAN_HEADING, constants.INIT_HUMAN_TRACKPOS)
        state = tuple2array(ob)

        while initial_speed < np.random.uniform(constants.MIN_INIT_SPEED, constants.MAX_INIT_SPEED):
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
                action = model.select_stochastic_action(state)
                # action = model.select_deterministic_action(state)
            else:
                action = model.select_action(state, memory, measurements_summary)

            action = np.clip(action, -1,1)
            ob, reward, done, current_pointer, _, _ = env.step(action, episode_step=t, pre_obs=t1_obs, t_2_obs=t2_obs, episode_num=i_episode-1)
            if args.algorithm == constants.ALGORITHM_TYPE.GAIL:
                reward = model.get_reward(state, action)
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
                    model.batch_size = time_step
                    last_value = model.policy_old.critic(state_eval)
                    model.update_actor_critic(memory, last_value)
                    if args.algorithm == constants.ALGORITHM_TYPE.GAIL:

                        # check criterion whether or not training the discriminator to avoid overfit:
                        states = torch.stack(memory.states).squeeze(dim=1).to(device).detach()
                        actions = torch.stack(memory.actions).squeeze(dim=1).to(device).detach()
                        learner_acc = ((model.discriminator(torch.cat([states, actions], dim=1)) > 0.5).float()).mean()
                        demonstrations = torch.Tensor(human_demonstrations)
                        expert_acc = ((model.discriminator(demonstrations) < 0.5).float()).mean()
                        logging.info("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
                        print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
                        if expert_acc < 0.95 or learner_acc > 0.85:
                            expert_acc, learner_acc = model.train_discriminator(memory, human_demonstrations)
                            logging.info("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
                            print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
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
                substraction = (args.max_length*2)%args.mini_batch_size
                print("substraction: ", episode_length)
                tmp = max(512, args.max_length*2 - substraction)
                update_timestep = min(tmp, args.max_timesteps) # 5.11.19
            print("update_timestep: ", update_timestep)



        writer.add_scalar('data/total_length', episode_length, i_episode)
        writer.add_scalar('data/total_reward', running_reward, i_episode)
        writer.add_scalar('var/steering_var', np.mean(measurements_summary.steering_var), i_episode)
        writer.add_scalar('var/throttle_var', np.mean(measurements_summary.throttle_var), i_episode)
        measurements_summary.clear_summary()

        total_episodes +=1
        print("total_episodes: ", total_episodes)
        print('Episode {} \t length: {} \t reward: {}'.format(i_episode, episode_length, running_reward))
        running_reward = 0
        episode_length = 0

        if args.train:
            save_model(model, episode_reward, best_score, args.algorithm)


if __name__ == '__main__':

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    K.set_session(sess)
    main()

