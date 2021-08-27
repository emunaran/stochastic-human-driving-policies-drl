from dataclasses import dataclass
import threading
import queue


@dataclass
class ServerConfig:
    counter: int
    game_ended: bool
    # time_step: int
    send_policy_action: bool
    initial_state: bool
    s_obs_id: int
    c_obs_id: int
    last_sample_time: float
    a: int
    # round_num: int
    # pre_episode_step: int
    condition: threading.Condition
    q: queue.Queue
