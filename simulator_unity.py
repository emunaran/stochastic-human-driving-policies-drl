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

counter = 0
sio = socketio.Server()
app = Flask(__name__)
condition = threading.Condition()
q = queue.Queue()
game_ended = False
time_step = 0
train = True
if train:
    max_episode_step = 4608
else:
    max_episode_step = 9999999
send_policy_action = False
initial_state = True
s_obs_id = 0
c_obs_id = 0
last_sample_time = time.time()
a = 0

round_num = 1
pre_episode_step = 0

# number of different samples in the array of the GPs
num_samples = 100

def step(u, episode_step, pre_obs, t_2_obs, episode_num):
    print("action is", u)
    print("episode step: ", episode_step)
    this_action = agent_to_unity(u)

    with condition:
        global send_policy_action
        while q.empty() and send_policy_action == False:
            # One-Step Dynamics Update #################################
            # Apply the Agent's action into Unity
            send_control(this_action)
            global a
            a += 1
            send_policy_action = True
            condition.notify_all()
            condition.wait()

    with condition:
        sample = q.get()
        global counter
        counter = counter + 1




    # Get the current full-observation from Unity

    speed = np.array(float(sample['speed']), dtype=np.float32)
    sign_theta = np.array(float(sample['sign_theta']), dtype=np.float32)*(np.pi/180)
    trackPos = np.array(float(sample['trackPos']), dtype=np.float32)
    collisionDetector = np.array(float(sample['collisionDetector']), dtype=np.float32)
    road_boundries = np.array(float(sample['trackPos']), dtype=np.float32)/6

    mileage = re.sub('[,]', '', sample['mileage'])
    pointer = re.sub('[,]', '', sample['pointer'])
    pointer = round(float(pointer))


    #human expert predictions
    training_road = np.array(float(sample['training_road']), dtype=np.float32)
    initial_sample_num = 97
    human = expert_training_road_1
    sample_number = (episode_num+initial_sample_num) % num_samples + 1

    human_arc_length_arr = human.iloc[:,0]

    mean_behavior = False
    if mean_behavior == True:
        human_trackPos_arr = human[['trackPos_mu']]
        human_trackPos_arr = (human_trackPos_arr.iloc[:,0])


        human_speed_arr = human[['speed_mu']]
        human_speed_arr = (human_speed_arr.iloc[:,0])

        human_heading_arr = human[['heading_mu']]
        human_heading_arr = (human_heading_arr.iloc[:,0])


    else:
        human_trackPos_arr = human[['trackPos_sample_{}'.format(sample_number)]]
        print('trackPos_sample_{}'.format(sample_number))
        human_trackPos_arr = (human_trackPos_arr.iloc[:,0])


        human_speed_arr = human[['speed_sample_{}'.format(sample_number)]]
        human_speed_arr = (human_speed_arr.iloc[:,0])

        human_heading_arr = human[['heading_sample_{}'.format(sample_number)]]
        human_heading_arr = (human_heading_arr.iloc[:,0])


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
    global pre_episode_step

    if ind <5 and np.abs(pre_episode_step - episode_step) > 100:
        global round_num
        round_num = round_num + 1
        print("round_num ", round_num)
        pre_episode_step = episode_step



    current_steering = u[0]
    current_torque = u[1]

    obs = make_observation(sample, current_steering,current_torque, pre_obs, t_2_obs, human_heading, human_trackPos)
    human_heading = human_heading*(np.pi/180)

    # Get the current full-observation from Unity
    steering_delta = np.abs(current_steering - pre_obs.current_steering)
    torque_delta = np.abs(current_torque - pre_obs.current_torque)

    trackPos_delta = np.abs(obs.trackPos - pre_obs.trackPos)
    sign_theta_delta = np.abs(obs.sign_theta - pre_obs.sign_theta)

    global time_step
    time_step += 1

    with_std = False
    if with_std:
        progress = (human_speed - np.abs(speed-human_speed)) * np.cos(sign_theta - human_heading)/human_std_heading - 10 * np.abs(trackPos - human_trackPos)/human_std_trackPos - 2 * speed * steering_delta - 10*torque_delta# - 2 * speed * np.max([0, np.abs(trackPos) - 0.5])
    else:
        progress = (human_speed - np.abs(speed-human_speed)) - 20 * np.abs(trackPos - human_trackPos) - 100 * steering_delta - 10*torque_delta # modification ver 4 22.10.19 23:30

    print("human_trackPos: ",trackPos - human_trackPos, " human_heading: ", sign_theta - human_heading, " human_speed: ", speed - human_speed)

    reward = progress

    # Termination judgement #########################
    episode_terminate = False

    if (np.abs(road_boundries) > 1 or (time_step > 20 and speed < 5) or np.abs(sign_theta*(np.pi/180)) > np.pi/2 or collisionDetector==1) and train:

        if np.abs(trackPos) > 6:
            obs = obs._replace(sensors_segments=(np.zeros(288)-1),
                               boundaries_sensors=(np.zeros(19)-1)
                               )
            print(obs)
        print("terminate")
        reward -= 100
        episode_terminate = True
        time_step = 0
    elif episode_step + 1 >= max_episode_step:
        print("max episode's steps termination")
        episode_terminate = True
        time_step = 0

    return obs, reward, episode_terminate, pointer, round_num, {}


def end():
    sio.emit("endGame", data={}, skip_sid=True)
    global game_ended
    game_ended = True
    global s_obs_id
    s_obs_id = 0
    global initial_state
    initial_state = True
    global a
    a=0
    with condition:
        while game_ended:
            condition.notify_all()
            condition.wait()


def get_init_obs():
    with condition:
        while q.empty():
            condition.notify_all()
            condition.wait()
    return q.get()


def agent_to_unity(u):
    steering_angle = u[0]

    if (u[1]>0):
        throttle = u[1]
        brake = 0
    else:
        throttle = 0
        brake = u[1]*(-1)

    unity_action = {'steering_angle': steering_angle, 'throttle': throttle, 'brake': brake}

    return unity_action


def make_observation(data, current_steering, current_torque, pre_obs, t_2_obs, human_heading, human_trackPos):
    right_obstacles_front = []
    right_obstacles_front_num = 1#10
    for i in range(right_obstacles_front_num):
        right_obstacles_front.append(float(data['rightObstacleFront{}'.format(i)]))

    right_obstacles_back = []
    right_obstacles_back_num = 1  # 10
    for i in range(right_obstacles_back_num):
        right_obstacles_back.append(float(data['rightObstacleBack{}'.format(i)]))

    left_obstacles_front = []
    left_obstacles_front_num = 1#10
    for i in range(left_obstacles_front_num):
        left_obstacles_front.append(float(data['leftObstacleFront{}'.format(i)]))

    left_obstacles_back = []
    left_obstacles_back_num = 1  # 10
    for i in range(left_obstacles_back_num):
        left_obstacles_back.append(float(data['leftObstacleBack{}'.format(i)]))

    boundaries_sensors = []
    boundaries_sensors_num = 19
    for i in range(boundaries_sensors_num):
        boundaries_sensors.append(float(data['boundaries{}'.format(i)]))

    curvatures = []
    curvatures_num = 19
    for i in range(curvatures_num):
        curvatures.append(float(data['carvature{}'.format(i)]))

    sensors_segments = []
    sensors_segments_num = 288
    for i in range(sensors_segments_num):
        sensors_segments.append(float(data['sensors_segments{}'.format(i)]))

    obstacle_detector = []
    obstacle_detector_num = 288
    for i in range(obstacle_detector_num):
        obstacle_detector.append(float(data['obsDetector{}'.format(i)]))

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
            #'road_boundries',
            'boundaries_sensors',
            'curvatures',
            'collisionDetector',
            #'boundaries_sensors',
            'sensors_segments',
            'obstacle_detector',
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
            'pre_curvatures',
            'pre_boundaries_sensors',
            'pre_sensors_segments',
            'pre_obstacle_detector',
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
            't_2_curvatures',
            't_2_boundaries_sensors',
            't_2_sensors_segments',
            't_2_obstacle_detector'

    ]

    Observation = col.namedtuple('Observation', names)

    obs = Observation(
        current_steering=np.array(current_steering, dtype=np.float32),
        current_torque=np.array(current_torque, dtype=np.float32),
        speed=np.clip(np.array(float(data['speed']), dtype=np.float32) / 100.0, 0, 1),
        sign_theta=np.clip(np.array(float(data['sign_theta']), dtype=np.float32)/90, -1, 1),#-human_heading
        trackPos=np.clip(np.array(float(data['trackPos']), dtype=np.float32) / 6., -1, 1), #-human_trackPos
        rightLane=np.array(float(data['rightLane']), dtype=np.float32) / 9,
        leftLane=np.array(float(data['leftLane']), dtype=np.float32) / 9,
        right_obstacles_front=np.clip(np.array(right_obstacles_front, dtype=np.float32) / 300, 0, 1),
        right_obstacles_back=np.clip(np.array(right_obstacles_back, dtype=np.float32) / 300, 0, 1),
        left_obstacles_front=np.clip(np.array(left_obstacles_front, dtype=np.float32) / 300, 0, 1),
        left_obstacles_back=np.clip(np.array(left_obstacles_back, dtype=np.float32) / 300, 0, 1),
        curvatures=np.clip(np.array(curvatures, dtype=np.float32), -1, 1),
        collisionDetector=np.clip(np.array(float(data['collisionDetector']), dtype=np.float32), 0, 1),
        boundaries_sensors=np.clip(np.array(boundaries_sensors, dtype=np.float32) / 200, -1, 1),
        sensors_segments=np.clip(np.array(sensors_segments, dtype=np.float32)/300, 0, 1),
        obstacle_detector= np.clip(np.array(obstacle_detector, dtype=np.float32), -1, 1),
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
        pre_curvatures=pre_obs.curvatures,
        pre_boundaries_sensors=pre_obs.boundaries_sensors,
        pre_sensors_segments=pre_obs.sensors_segments,
        pre_obstacle_detector=pre_obs.obstacle_detector,
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
        t_2_curvatures=t_2_obs.curvatures,
        t_2_boundaries_sensors=t_2_obs.boundaries_sensors,
        t_2_sensors_segments=t_2_obs.sensors_segments,
        t_2_obstacle_detector= t_2_obs.obstacle_detector


        )

    return obs

def make_pre_observation(data, current_steering, current_torque):
    right_obstacles_front = []
    right_obstacles_front_num = 1  # 10
    for i in range(right_obstacles_front_num):
        right_obstacles_front.append(float(data['rightObstacleFront{}'.format(i)]))

    right_obstacles_back = []
    right_obstacles_back_num = 1  # 10
    for i in range(right_obstacles_back_num):
        right_obstacles_back.append(float(data['rightObstacleBack{}'.format(i)]))

    left_obstacles_front = []
    left_obstacles_front_num = 1  # 10
    for i in range(left_obstacles_front_num):
        left_obstacles_front.append(float(data['leftObstacleFront{}'.format(i)]))

    left_obstacles_back = []
    left_obstacles_back_num = 1  # 10
    for i in range(left_obstacles_back_num):
        left_obstacles_back.append(float(data['leftObstacleBack{}'.format(i)]))

    boundaries_sensors = []
    boundaries_sensors_num = 19
    for i in range(boundaries_sensors_num):
        boundaries_sensors.append(float(data['boundaries{}'.format(i)]))

    curvatures = []
    curvatures_num = 19
    for i in range(curvatures_num):
        curvatures.append(float(data['carvature{}'.format(i)]))

    sensors_segments = []
    sensors_segments_num = 288
    for i in range(sensors_segments_num):
        sensors_segments.append(float(data['sensors_segments{}'.format(i)]))

    obstacle_detector = []
    obstacle_detector_num = 288
    for i in range(obstacle_detector_num):
        obstacle_detector.append(float(data['obsDetector{}'.format(i)]))

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
            'curvatures',
            'boundaries_sensors',
            'sensors_segments',
            'obstacle_detector'

    ]

    Observation = col.namedtuple('Observation', names)

    pre_obs = Observation(
        current_steering=np.array(current_steering, dtype=np.float32),
        current_torque=np.array(current_torque, dtype=np.float32),
        speed=np.clip(np.array(float(data['speed']), dtype=np.float32) / 100.0, 0, 1),
        sign_theta=np.clip(np.array(float(data['sign_theta']), dtype=np.float32) / 90, -1, 1),
        trackPos=np.clip(np.array(float(data['trackPos']), dtype=np.float32) / 6., -1, 1),
        rightLane=np.array(float(data['rightLane']), dtype=np.float32) / 9,
        leftLane=np.array(float(data['leftLane']), dtype=np.float32) / 9,
        right_obstacles_front=np.clip(np.array(right_obstacles_front, dtype=np.float32) / 300, 0, 1),
        right_obstacles_back=np.clip(np.array(right_obstacles_back, dtype=np.float32) / 300, 0, 1),
        left_obstacles_front=np.clip(np.array(left_obstacles_front, dtype=np.float32) / 300, 0, 1),
        left_obstacles_back=np.clip(np.array(left_obstacles_back, dtype=np.float32) / 300, 0, 1),
        curvatures=np.clip(np.array(curvatures, dtype=np.float32), -1, 1),
        boundaries_sensors=np.clip(np.array(boundaries_sensors, dtype=np.float32) / 200, -1, 1),
        sensors_segments=np.clip(np.array(sensors_segments, dtype=np.float32) / 300, 0, 1),
        obstacle_detector= np.clip(np.array(obstacle_detector, dtype=np.float32), -1, 1)
        )

    return pre_obs


def send_control(this_action):
    sio.emit(
        "steer",
        data={
            'steering_angle': this_action.get("steering_angle").__str__(),
            'throttle': this_action.get("throttle").__str__(),
            'brake': this_action.get("brake").__str__()
        },
        skip_sid=True)
    global s_obs_id
    s_obs_id += 1



def runSocket():
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, Flask(__name__))

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


@sio.on('telemetry')
def set_obs(sid, obs):

    global send_policy_action
    with condition:
        global initial_state
        while initial_state == True:
            if q.empty() == False:
                q.get()
                send_signal()
            q.put(obs)
            send_signal()
            initial_state = False
            break
        while send_policy_action == False:
            if np.array(np.abs(float(obs['trackPos'])))<6:
                global game_ended
                game_ended = False
            send_signal()
            condition.notify_all()
            condition.wait()
        send_signal()

    with condition:
        while q.empty() and send_policy_action == True:

            # global game_ended
            if game_ended and np.array(np.abs(float(obs['trackPos']))) > 6:
                # print("still ended")
                break

            else:
                game_ended = False
                obs['c_obs_id'] = re.sub('[,]', '', obs['c_obs_id'])
                global c_obs_id
                c_obs_id = int(np.array(float(obs['c_obs_id']), dtype=np.float32))

                if s_obs_id == c_obs_id:
                    global last_sample_time
                    elapsed_time = time.time() - last_sample_time
                    if elapsed_time > 0.1:
                        q.put(obs)
                        last_sample_time = time.time()
                        send_signal()
                        break

                else:
                    send_signal()
                    break

                    #raise Exception('client and server observation not sharing the same id')

    with condition:
        while not q.empty() and game_ended == False:
            send_signal()
            send_policy_action = False
            condition.notify_all()
            condition.wait()
        send_signal()


def send_signal():
    sio.emit('', data={}, skip_sid=True)



if __name__:



    expert_training_road_1 = pd.read_csv("data/expert/gp/GP_expert.csv")
    # expert_training_road_1 = pd.read_csv("ProPMs/expert_training_road_1.csv")
    # expert_training_road_2 = pd.read_csv("ProPMs/expert_training_road_2.csv")

    t1 = threading.Thread(target=runSocket, name='thread1')
    t1.start()

