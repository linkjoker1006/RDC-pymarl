import numpy as np


class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))


class FlatThenDecayThenFlatSchedule():

    def __init__(self,
                 start_value,
                 finish_value,
                 start_time,
                 end_time,
                 decay="exp"):

        self.start_value = start_value
        self.finish_value = finish_value
        self.start_time = start_time
        self.end_time = end_time
        self.delta = (self.start_value - self.finish_value) / (self.end_time - self.start_time)
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * (self.end_time - self.start_time) / np.log(self.finish_value) if self.finish_value > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            if T <= self.start_time:
                return self.start_value
            elif T <= self.end_time:
                return max(self.finish_value, self.start_value - self.delta * (T - self.start_time))
            else:
                return self.finish_value
        elif self.decay in ["exp"]:
            if T <= self.start_time:
                return self.start_value
            elif T <= self.end_time:
                return  min(self.start_value, max(self.finish_value, np.exp(- (T - self.start_time) / self.exp_scaling)))
            else:
                return self.finish_value


class LinearIncreaseSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length

    def eval(self, T):
        return min(self.finish, self.start - self.delta * T)
