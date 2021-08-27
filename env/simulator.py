import gym
import threading
import collections as col
import re
import pandas as pd
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
import time
import queue
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

        # Get the current full-observation from Unity
        speed = np.array(float(sample['speed']), dtype=np.float32)
        sign_theta = np.array(float(sample['sign_theta']), dtype=np.float32) * (np.pi / 180)
        trackPos = np.array(float(sample['trackPos']), dtype=np.float32)
        collisionDetector = np.array(float(sample['collisionDetector']), dtype=np.float32)
        road_boundries = np.array(float(sample['trackPos']), dtype=np.float32) / 6

        mileage = re.sub('[,]', '', sample['mileage'])
        pointer = re.sub('[,]', '', sample['pointer'])
        pointer = round(float(pointer))

        # human expert predictions
        training_road = np.array(float(sample['training_road']), dtype=np.float32)
        initial_sample_num = 0
        human = self.expert_gp_samples_training_road
        sample_number = (episode_num + initial_sample_num) % self.num_samples + 1

        human_arc_length_arr = human.iloc[:, 0]

        mean_behavior = True
        if mean_behavior == True:
            human_trackPos_arr = human[['trackPos_mu']]
            human_trackPos_arr = (human_trackPos_arr.iloc[:, 0])

            human_speed_arr = human[['speed_mu']]
            human_speed_arr = (human_speed_arr.iloc[:, 0])

            human_heading_arr = human[['heading_mu']]
            human_heading_arr = (human_heading_arr.iloc[:, 0])


        else:
            human_trackPos_arr = human[['trackPos_sample_{}'.format(sample_number)]]
            print('trackPos_sample_{}'.format(sample_number))
            human_trackPos_arr = (human_trackPos_arr.iloc[:, 0])

            human_speed_arr = human[['speed_sample_{}'.format(sample_number)]]
            human_speed_arr = (human_speed_arr.iloc[:, 0])

            human_heading_arr = human[['heading_sample_{}'.format(sample_number)]]
            human_heading_arr = (human_heading_arr.iloc[:, 0])

        # standard deviation

        human_std_trackPos_arr = human[['trackPos_std']]
        human_std_trackPos_arr = (human_std_trackPos_arr.iloc[:, 0])

        human_std_speed_arr = human[['speed_std']]
        human_std_speed_arr = (human_std_speed_arr.iloc[:, 0])

        human_std_heading_arr = human[['heading_std']]
        human_std_heading_arr = (human_std_heading_arr.iloc[:, 0])

        #
        ind = round(float(mileage))
        max_human_mileage = np.max(human_arc_length_arr)
        if ind > max_human_mileage:
            ind = max_human_mileage - 1
        human_trackPos = human_trackPos_arr[ind]
        human_heading = human_heading_arr[ind]
        human_speed = human_speed_arr[ind]
        human_std_trackPos = human_std_trackPos_arr[ind]
        human_std_heading = human_std_heading_arr[ind]
        human_std_speed = human_std_speed_arr[ind]

        # print round number
        # global pre_episode_step

        if ind < 5 and np.abs(self.pre_episode_step - episode_step) > 100:
            # global round_num
            self.round_num = self.round_num + 1
            print("round_num ", self.round_num)
            self.pre_episode_step = episode_step

        current_steering = u[0]
        current_torque = u[1]

        obs = env_utils.make_observation(sample, current_steering, current_torque, pre_obs, t_2_obs, human_heading,
                               human_trackPos)
        human_heading = human_heading * (np.pi / 180)

        # Get the current full-observation from Unity
        steering_delta = np.abs(current_steering - pre_obs.current_steering)
        torque_delta = np.abs(current_torque - pre_obs.current_torque)

        trackPos_delta = np.abs(obs.trackPos - pre_obs.trackPos)
        sign_theta_delta = np.abs(obs.sign_theta - pre_obs.sign_theta)

        # global time_step
        self.time_step += 1

        with_std = False
        if with_std:
            progress = (human_speed - np.abs(speed - human_speed)) * np.cos(
                sign_theta - human_heading) / human_std_heading - 10 * np.abs(
                trackPos - human_trackPos) / human_std_trackPos - 2 * speed * steering_delta - 10 * torque_delta  # - 2 * speed * np.max([0, np.abs(trackPos) - 0.5])
        else:
            progress = (human_speed - np.abs(speed - human_speed)) - 20 * np.abs(
                trackPos - human_trackPos) - 100 * steering_delta - 10 * torque_delta  # modification ver 4 22.10.19 23:30

        print("human_trackPos: ", trackPos - human_trackPos, " human_heading: ", sign_theta - human_heading,
              " human_speed: ", speed - human_speed)

        reward = progress

        # Termination judgement #########################
        episode_terminate = False

        if (np.abs(road_boundries) > 1 or (self.time_step > 20 and speed < 5) or np.abs(
                sign_theta * (np.pi / 180)) > np.pi / 2 or collisionDetector == 1) and self.train:

            if np.abs(trackPos) > 6:
                obs = obs._replace(sensors_segments=(np.zeros(288) - 1),
                                   boundaries_sensors=(np.zeros(19) - 1)
                                   )
                print(obs)
            print("terminate")
            reward -= 100
            episode_terminate = True
            self.time_step = 0
        elif episode_step + 1 >= self.max_episode_step and self.train:
            print("max episode's steps termination")
            episode_terminate = True
            self.time_step = 0

        return obs, reward, episode_terminate, pointer, self.round_num, {}

