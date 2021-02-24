import collections as col
import pandas as pd
import numpy as np
import copy
from utils.utils import tuple2array, get_t1_obs, get_t2_obs

def main():

    state_demonstrations = []
    df = pd.read_csv('data/expert/demonstrations/driving_log_recovered_w_2d_actions.csv')
    for index, obs in df.iterrows():

        if index==0:
            t1_obs = make_pre_observation(obs)
            t2_obs = copy.copy(t1_obs)

        ob = make_observation(obs, t1_obs, t2_obs)
        state = tuple2array(ob)
        state_demonstrations.append(state)
        t1_obs = get_t1_obs(t1_obs, ob)
        t2_obs = get_t2_obs(t2_obs, ob)
        print(index)

    state_demonstrations_df = pd.DataFrame(state_demonstrations)
    state_demonstrations_df.to_csv("./data/expert/demonstrations/state_demo.csv", index=False)


def make_observation(data, pre_obs, t_2_obs):
    right_obstacles_front = []
    right_obstacles_front_num = 1  # 10
    for i in range(right_obstacles_front_num):
        right_obstacles_front.append(float(data['RightObstaclesFront[{}]'.format(i)]))

    right_obstacles_back = []
    right_obstacles_back_num = 1  # 10
    for i in range(right_obstacles_back_num):
        right_obstacles_back.append(float(data['RightObstaclesBack[{}]'.format(i)]))

    left_obstacles_front = []
    left_obstacles_front_num = 1  # 10
    for i in range(left_obstacles_front_num):
        left_obstacles_front.append(float(data['LeftObstaclesFront[{}]'.format(i)]))

    left_obstacles_back = []
    left_obstacles_back_num = 1  # 10
    for i in range(left_obstacles_back_num):
        left_obstacles_back.append(float(data['LeftObstaclesBack[{}]'.format(i)]))

    sensors_segments = []
    sensors_segments_num = 288
    for i in range(sensors_segments_num):
        sensors_segments.append(float(data['SensorSegmentDist[{}]'.format(i)]))

    names = [
            'current_steering',
            'current_torque',
            'speed',
            'sign_theta',
            'trackPos',
            'rightLane',
            'leftLane',
            'right_obstacles_front',
            'right_obstacles_back',
            'left_obstacles_front',
            'left_obstacles_back',
            'sensors_segments',
            'pre_steering',
            'pre_torque',
            'pre_speed',
            'pre_sign_theta',
            'pre_trackPos',
            'pre_rightLane',
            'pre_leftLane',
            'pre_right_obstacles_front',
            'pre_right_obstacles_back',
            'pre_left_obstacles_front',
            'pre_left_obstacles_back',
            'pre_sensors_segments',
            't_2_steering',
            't_2_torque',
            't_2_speed',
            't_2_sign_theta',
            't_2_trackPos',
            't_2_rightLane',
            't_2_leftLane',
            't_2_right_obstacles_front',
            't_2_right_obstacles_back',
            't_2_left_obstacles_front',
            't_2_left_obstacles_back',
            't_2_sensors_segments',

    ]

    Observation = col.namedtuple('Observation', names)

    obs = Observation(
        current_steering=np.array(float(data['steeringAngle']), dtype=np.float32),
        current_torque=np.array(float(data['torque']), dtype=np.float32),
        speed=np.clip(np.array(float(data['speed']), dtype=np.float32) / 100.0, 0, 1),
        sign_theta=np.clip(np.array(float(data['signedTheta']), dtype=np.float32)/90, -1, 1),#-human_heading
        trackPos=np.clip(np.array(float(data['trackPos']), dtype=np.float32) / 6., -1, 1), #-human_trackPos
        rightLane=np.array(float(data['trackPos'])+3, dtype=np.float32) / 9,
        leftLane=np.array(float(data['trackPos'])-3, dtype=np.float32) / 9,
        right_obstacles_front=np.clip(np.array(right_obstacles_front, dtype=np.float32) / 300, 0, 1),
        right_obstacles_back=np.clip(np.array(right_obstacles_back, dtype=np.float32) / 300, 0, 1),
        left_obstacles_front=np.clip(np.array(left_obstacles_front, dtype=np.float32) / 300, 0, 1),
        left_obstacles_back=np.clip(np.array(left_obstacles_back, dtype=np.float32) / 300, 0, 1),
        sensors_segments=np.clip(np.array(sensors_segments, dtype=np.float32)/300, 0, 1),
        pre_steering=pre_obs.current_steering,
        pre_torque=pre_obs.current_torque,
        pre_speed=pre_obs.speed,
        pre_sign_theta=pre_obs.sign_theta,
        pre_trackPos=pre_obs.trackPos,
        pre_rightLane=pre_obs.rightLane,
        pre_leftLane=pre_obs.leftLane,
        pre_right_obstacles_front=pre_obs.right_obstacles_front,
        pre_right_obstacles_back=pre_obs.right_obstacles_back,
        pre_left_obstacles_front=pre_obs.left_obstacles_front,
        pre_left_obstacles_back=pre_obs.left_obstacles_back,
        pre_sensors_segments=pre_obs.sensors_segments,
        t_2_steering=t_2_obs.current_steering,
        t_2_torque=t_2_obs.current_torque,
        t_2_speed=t_2_obs.speed,
        t_2_sign_theta=t_2_obs.sign_theta,
        t_2_trackPos=t_2_obs.trackPos,
        t_2_rightLane=t_2_obs.rightLane,
        t_2_leftLane=t_2_obs.leftLane,
        t_2_right_obstacles_front=t_2_obs.right_obstacles_front,
        t_2_right_obstacles_back=t_2_obs.right_obstacles_back,
        t_2_left_obstacles_front=t_2_obs.left_obstacles_front,
        t_2_left_obstacles_back=t_2_obs.left_obstacles_back,
        t_2_sensors_segments=t_2_obs.sensors_segments,


        )

    return obs

def make_pre_observation(data):
    right_obstacles_front = []
    right_obstacles_front_num = 1  # 10
    for i in range(right_obstacles_front_num):
        right_obstacles_front.append(float(data['RightObstaclesFront[{}]'.format(i)]))

    right_obstacles_back = []
    right_obstacles_back_num = 1  # 10
    for i in range(right_obstacles_back_num):
        right_obstacles_back.append(float(data['RightObstaclesBack[{}]'.format(i)]))

    left_obstacles_front = []
    left_obstacles_front_num = 1  # 10
    for i in range(left_obstacles_front_num):
        left_obstacles_front.append(float(data['LeftObstaclesFront[{}]'.format(i)]))

    left_obstacles_back = []
    left_obstacles_back_num = 1  # 10
    for i in range(left_obstacles_back_num):
        left_obstacles_back.append(float(data['LeftObstaclesBack[{}]'.format(i)]))

    sensors_segments = []
    sensors_segments_num = 288
    for i in range(sensors_segments_num):
        sensors_segments.append(float(data['SensorSegmentDist[{}]'.format(i)]))

    names = [
            'current_steering',
            'current_torque',
            'speed',
            'sign_theta',
            'trackPos',
            'rightLane',
            'leftLane',
            'right_obstacles_front',
            'right_obstacles_back',
            'left_obstacles_front',
            'left_obstacles_back',
            'sensors_segments',

    ]

    Observation = col.namedtuple('Observation', names)

    pre_obs = Observation(
        current_steering=np.array(float(data['steeringAngle']), dtype=np.float32),
        current_torque=np.array(float(data['torque']), dtype=np.float32),
        speed=np.clip(np.array(float(data['speed']), dtype=np.float32) / 100.0, 0, 1),
        sign_theta=np.clip(np.array(float(data['signedTheta']), dtype=np.float32) / 90, -1, 1),
        trackPos=np.clip(np.array(float(data['trackPos']), dtype=np.float32) / 6., -1, 1),
        rightLane=np.array(float(data['trackPos'])+3, dtype=np.float32) / 9,
        leftLane=np.array(float(data['trackPos'])-3, dtype=np.float32) / 9,
        right_obstacles_front=np.clip(np.array(right_obstacles_front, dtype=np.float32) / 300, 0, 1),
        right_obstacles_back=np.clip(np.array(right_obstacles_back, dtype=np.float32) / 300, 0, 1),
        left_obstacles_front=np.clip(np.array(left_obstacles_front, dtype=np.float32) / 300, 0, 1),
        left_obstacles_back=np.clip(np.array(left_obstacles_back, dtype=np.float32) / 300, 0, 1),
        sensors_segments=np.clip(np.array(sensors_segments, dtype=np.float32) / 300, 0, 1),
        )

    return pre_obs

if __name__ == '__main__':
    main()