

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.masks = []
        self.values = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.masks[:]
        del self.values[:]


class MeasurementsSummary:
    def __init__(self):
        self.steering_var = []
        self.throttle_var = []

    def clear_summary(self):
        del self.steering_var[:]
        del self.throttle_var[:]