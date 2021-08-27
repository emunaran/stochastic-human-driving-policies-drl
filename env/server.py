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
from env.server_config import ServerConfig


sio = socketio.Server()
app = Flask(__name__)
server = ServerConfig(counter=0,
                      game_ended=False,
                      # time_step=0,
                      send_policy_action=False,
                      initial_state=True,
                      s_obs_id=0,
                      c_obs_id=0,
                      last_sample_time=time.time(),
                      a=0,
                      # round_num=1,
                      # pre_episode_step=0,
                      condition=threading.Condition(),
                      q=queue.Queue())


def send_action_get_next_state(action):
    with server.condition:
        # global send_policy_action
        while server.q.empty() and server.send_policy_action == False:
            # One-Step Dynamics Update #################################
            # Apply the Agent's action into Unity
            send_control(action)
            # global a
            server.a += 1
            server.send_policy_action = True
            server.condition.notify_all()
            server.condition.wait()

    with server.condition:
        sample = server.q.get()
        # global counter
        server.counter = server.counter + 1

    return sample


def end():
    sio.emit("endGame", data={}, skip_sid=True)
    # global game_ended
    server.game_ended = True
    # global s_obs_id
    server.s_obs_id = 0
    # global initial_state
    server.initial_state = True
    # global a
    server.a = 0
    with server.condition:
        while server.game_ended:
            server.condition.notify_all()
            server.condition.wait()


def get_init_obs():
    with server.condition:
        while server.q.empty():
            server.condition.notify_all()
            server.condition.wait()
    return server.q.get()


def send_control(this_action):
    sio.emit(
        "steer",
        data={
            'steering_angle': this_action.get("steering_angle").__str__(),
            'throttle': this_action.get("throttle").__str__(),
            'brake': this_action.get("brake").__str__()
        },
        skip_sid=True)
    # global s_obs_id
    server.s_obs_id += 1


def run_socket():
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, Flask(__name__))

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


@sio.on('telemetry')
def set_obs(sid, obs):

    # global send_policy_action
    with server.condition:
        # global initial_state
        while server.initial_state == True:
            if server.q.empty() == False:
                server.q.get()
                send_signal()
            server.q.put(obs)
            send_signal()
            server.initial_state = False
            break
        while server.send_policy_action == False:
            if np.array(np.abs(float(obs['trackPos'])))<6:
                # global game_ended
                server.game_ended = False
            send_signal()
            server.condition.notify_all()
            server.condition.wait()
        send_signal()

    with server.condition:
        while server.q.empty() and server.send_policy_action==True:

            # global game_ended
            if server.game_ended and np.array(np.abs(float(obs['trackPos']))) > 6:
                # print("still ended")
                break

            else:
                server.game_ended = False
                obs['c_obs_id'] = re.sub('[,]', '', obs['c_obs_id'])
                # global c_obs_id
                server.c_obs_id = int(np.array(float(obs['c_obs_id']), dtype=np.float32))

                if server.s_obs_id == server.c_obs_id:
                    # global last_sample_time
                    elapsed_time = time.time() - server.last_sample_time
                    if elapsed_time > 0.1:
                        server.q.put(obs)
                        server.last_sample_time = time.time()
                        send_signal()
                        break

                else:
                    send_signal()
                    break

                    #raise Exception('client and server observation not sharing the same id')

    with server.condition:
        while not server.q.empty() and server.game_ended == False:
            send_signal()
            server.send_policy_action = False
            server.condition.notify_all()
            server.condition.wait()
        send_signal()


def send_signal():
    sio.emit('', data={}, skip_sid=True)


# if __name__ == '__main__':
def run():
    t1 = threading.Thread(target=run_socket, name='thread1')
    t1.start()
