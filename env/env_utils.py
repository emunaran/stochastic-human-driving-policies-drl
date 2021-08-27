import numpy as np
import collections as col


def agent_to_unity(u):
    steering_angle = u[0]

    if u[1] > 0:
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