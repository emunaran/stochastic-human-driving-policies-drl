

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
        self.alpha_1 = []
        self.alpha_2 = []
        self.alpha_3 = []

    def clear_summary(self):
        del self.steering_var[:]
        del self.throttle_var[:]
        del self.alpha_1[:]
        del self.alpha_2[:]
        del self.alpha_3[:]
