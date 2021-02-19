import socketio
from flask import Flask
import eventlet
import eventlet.wsgi

sio = socketio.Server()
# app = Flask(__name__)


@sio.on('telemetry')
def set_obs(sid, obs):
    print(f"obs: {obs}")

# def runSocket():
#     # wrap Flask application with engineio's middleware
#     app = socketio.Middleware(sio, Flask(__name__))
#
#     # deploy as an eventlet WSGI server
#     eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), app)

if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, Flask(__name__))

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), app)
    # runSocket()
#

