import pandas as pd
import numpy as np
from env import env_utils
from env.server import send_action_get_next_state, end
from env import server


class Simulator:
    def __init__(self, train, algorithm):
        self.num_samples = 100
        self.time_step = 0
        self.pre_episode_step = 0
        self.round_num = 1
        self.train = train
        self.max_episode_step = 4608
        self.expert_gp_samples_training_road = pd.read_csv("data/expert/gp/GP_expert.csv")
        self.algorithm = algorithm
        server.run()

    def end(self):
        end()

    def step(self, u, episode_step, pre_obs, t_2_obs, episode_num):

        print("action is", u)
        print("episode step: ", episode_step)

        this_action = env_utils.agent_to_unity(u)
        sample = send_action_get_next_state(this_action)

        mileage, pointer = env_utils.get_mileage_and_pointer(sample)

        # human expert predictions
        initial_sample_num = 0  # In case you want to restart training from a specific reference sample
        sample_number = (episode_num + initial_sample_num) % self.num_samples + 1

        human_trackPos, human_heading, human_speed, inx = env_utils.get_expert_reference(
                                                                         mileage,
                                                                         self.expert_gp_samples_training_road,
                                                                         self.algorithm,
                                                                         sample_number)

        complete_round_flag = inx < 5 and np.abs(self.pre_episode_step - episode_step) > 100
        if complete_round_flag:
            self.round_num = self.round_num + 1
            print("round_num ", self.round_num)
            self.pre_episode_step = episode_step

        current_steering = u[0]
        current_torque = u[1]

        obs = env_utils.make_observation(sample,
                                         current_steering,
                                         current_torque,
                                         pre_obs,
                                         t_2_obs,
                                         human_heading,
                                         human_trackPos)

        # trackPos_delta = np.abs(obs.trackPos - pre_obs.trackPos)
        # sign_theta_delta = np.abs(obs.sign_theta - pre_obs.sign_theta)
        # human_heading = human_heading * (np.pi / 180)

        # ---------- calculate immediate reward ---------- #
        speed, track_pos = env_utils.get_kinematic_features(sample)
        steering_delta = np.abs(current_steering - pre_obs.current_steering)
        torque_delta = np.abs(current_torque - pre_obs.current_torque)

        reward = (human_speed - np.abs(speed - human_speed)) - 20 * np.abs(track_pos - human_trackPos) - 100 * steering_delta - 10 * torque_delta
        print(f'trackPos err: {track_pos - human_trackPos}, speed err: {speed - human_speed}')
        self.time_step += 1

        # ---------- check termination states ---------- #
        obs, reward, episode_terminate = self.__check_termination(sample, speed, episode_step, reward, obs)

        return obs, reward, episode_terminate, pointer, self.round_num, {}

    def __check_termination(self, sample, speed, episode_step, reward, obs):
        episode_terminate = False
        sign_theta, collision_detector, road_boundaries = env_utils.get_termination_state_features(sample)

        exceed_road_boundaries = np.abs(road_boundaries) > 1
        drive_slow = self.time_step > 20 and speed < 5
        drive_opposite_direction = np.abs(sign_theta * (np.pi / 180)) > np.pi / 2
        pass_max_episode_steps = episode_step + 1 >= self.max_episode_step

        if (exceed_road_boundaries or drive_slow or drive_opposite_direction or collision_detector == 1) and self.train:
            print("terminate")
            if exceed_road_boundaries:
                obs = env_utils.flat_sensors_values(obs)
            reward -= 100
            episode_terminate = True
            self.time_step = 0
        elif pass_max_episode_steps and self.train:
            print("max episode's steps termination")
            episode_terminate = True
            self.time_step = 0

        return obs, reward, episode_terminate

