import os

os.sys.path.insert(0, os.path.abspath("../.."))
import numpy as np

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

    def update_policy(self):
        pass

    def eval(self):
        pass

    def cuda(self):
        pass

    def observe(self, r_t, s_t, s_t1, a_t, done):
        pass

    def random_action(self):
        action = np.random.uniform(self.lbound, self.rbound, self.nb_actions)
        return action

    def select_action(self, s_t, episode, decay_epsilon = True):
        action = np.random.uniform(self.lbound, self.rbound, self.nb_actions)
        return action

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
