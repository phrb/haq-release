import os

os.sys.path.insert(0, os.path.abspath("../.."))
import numpy as np
import pandas as pd

from lib.rl.memory import SequentialMemory

class RS(object):
    def __init__(self, nb_states, nb_actions, args):
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_actions = nb_actions
        self.is_training = True

        self.lbound = 0.  # args.lbound
        self.rbound = 1.  # args.rbound

        self.memory = SequentialMemory(limit = args.rmsize,
                                       window_length = args.window_length)

        self.design = pd.read_csv("experimental_designs/sobol_resnet50_600_samples.csv")
        self.design["Accuracy"] = float("inf")

        self.current_column = 0
        self.current_row = 0

        self.past_episode = 0

        self.current_action = float("inf")

    def update_policy(self):
        pass

    def eval(self):
        pass

    def cuda(self):
        pass

    def observe(self, r_t, s_t, s_t1, a_t, done):
        pass

    def random_action(self):
        self.current_action = self.design.iat[self.current_row, self.current_column]

        self.current_column += 1
        if self.current_column >= self.design.shape[1]:
            self.current_column = 0
            self.current_row += 1

        return self.current_action

    def select_action(self, s_t, episode, decay_epsilon = True):
        if episode != 0 and episode == self.past_episode:
            return self.current_action
        else:
            self.past_episode = episode
            return self.random_action()

    def save_accuracy(self, accuracy):
        self.design.at[(self.current_row - 1), "Accuracy"] = accuracy
        self.design.to_csv("sobol_resnet50_600_samples_results.csv", index = False)

    def reset(self, obs):
        pass

    def load_weights(self, output):
        pass

    def save_model(self, output):
        pass

    def seed(self, s):
        pass

    def soft_update(self, target, source):
        pass

    def hard_update(self, target, source):
        pass

    def get_delta(self):
        return np.array([0.0])

    def get_value_loss(self):
        return np.array([0.0])

    def get_policy_loss(self):
        return np.array([0.0])
