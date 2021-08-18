import os.path

import numpy as np
import torch
from pathlib import Path
import datetime

def tuple2array(ob):
    converted_obs = np.hstack((ob.sign_theta, ob.trackPos, ob.leftLane, ob.rightLane, ob.speed, ob.current_steering, ob.current_torque,
                               ob.sensors_segments, ob.left_obstacles_front,ob.left_obstacles_back, ob.right_obstacles_front, ob.right_obstacles_back, #ob.curvatures,
                               ob.pre_sign_theta, ob.pre_trackPos, ob.pre_leftLane, ob.pre_rightLane, ob.pre_speed, ob.pre_steering, ob.pre_torque,
                               ob.pre_left_obstacles_front,ob.pre_left_obstacles_back, ob.pre_right_obstacles_front, ob.pre_right_obstacles_back, #ob.pre_curvatures
                               ob.t_2_sign_theta, ob.t_2_trackPos, ob.t_2_leftLane, ob.t_2_rightLane, ob.t_2_speed, ob.t_2_steering, ob.t_2_torque,
                               ob.t_2_left_obstacles_front, ob.t_2_left_obstacles_back,ob.t_2_right_obstacles_front, ob.t_2_left_obstacles_back #ob.t_2_boundaries_sensors# ,
                                  ))
    return converted_obs


def get_t1_obs(t1_obs, ob):
    t1_obs = t1_obs._replace(
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
        # curvatures=ob.curvatures,
        # boundaries_sensors=ob.boundaries_sensors,
        sensors_segments=ob.sensors_segments,
        # obstacle_detector=ob.obstacle_detector
    )

    return t1_obs

def get_t2_obs(t2_obs, ob):
    t2_obs = t2_obs._replace(
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
        # curvatures=ob.curvatures,
        # boundaries_sensors=ob.pre_boundaries_sensors,
        sensors_segments=ob.pre_sensors_segments,
        # obstacle_detector=ob.pre_obstacle_detector

    )

    return t2_obs


def save_model(model, episode_reward, best_score, algorithm_name):

    if episode_reward > best_score:
        best_score = episode_reward
        # save the best model:
        torch.save(model.policy.actor.state_dict(), f"saved_models/{algorithm_name}/best/actor.pt")
        torch.save(model.policy.critic.state_dict(), f"saved_models/{algorithm_name}/best/critic.pt")

        torch.save(model.policy_old.actor.state_dict(), f"saved_models/{algorithm_name}/best/actor_old.pt")
        torch.save(model.policy_old.critic.state_dict(), f"saved_models/{algorithm_name}/best/critic_old.pt")

        if algorithm_name == 'GAIL':
            torch.save(model.discriminator.state_dict(), f"saved_models/{algorithm_name}/best/discriminator.pt")

    # save the last model:
    torch.save(model.policy.actor.state_dict(), f"saved_models/{algorithm_name}/actor.pt")
    torch.save(model.policy.critic.state_dict(), f"saved_models/{algorithm_name}/critic.pt")

    torch.save(model.policy_old.actor.state_dict(), f"saved_models/{algorithm_name}/actor_old.pt")
    torch.save(model.policy_old.critic.state_dict(), f"saved_models/{algorithm_name}/critic_old.pt")

    if algorithm_name == 'GAIL':
        torch.save(model.discriminator.state_dict(), f"saved_models/{algorithm_name}/discriminator.pt")


def save_model_checkpoint(model, algorithm_name):

    cp_path = f'saved_models/{algorithm_name}/checkpoints/{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
    Path(cp_path).mkdir(parents=True, exist_ok=True)

    torch.save(model.policy.actor.state_dict(), os.path.join(cp_path, 'actor.pt'))
    torch.save(model.policy.critic.state_dict(), os.path.join(cp_path, 'critic.pt'))

    torch.save(model.policy_old.actor.state_dict(), os.path.join(cp_path, 'actor_old.pt'))
    torch.save(model.policy_old.critic.state_dict(), os.path.join(cp_path, 'critic_old.pt'))

    if algorithm_name == 'GAIL':
        torch.save(model.discriminator.state_dict(), os.path.join(cp_path, 'discriminator.pt'))