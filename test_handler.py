#!/usr/bin/env python3
# import json
# import random

from viz_task45 import SailonViz as SViz
from viz_task123 import SailonViz as TViz
import numpy as np


class TestHandler:

    # Init function, accepts connection and address info
    def __init__(self, seed_list, domain: str = 'cartpole', novelty: int = 0, difficulty: str = 'easy',
                 seed: int = 123, trial_novelty: int = 0, day_offset: int = 0, use_img: bool = False,
                 path: str = "env_generator/envs/", use_gui: bool = False, check: bool = False, tdir="4"):

        # Set parameters
        self.seed = seed
        self.domain = domain
        self.novelty = novelty
        self.difficulty = 'easy'
        self.check = check
        self.trial_novelty = trial_novelty
        self.tdir = int(tdir)
        self.day_offset = day_offset
        self.use_img = use_img
        self.path = path
        self.use_gui = use_gui
        self.use_mock = 0
        self.use_novel = 0
        self.level = 0

        if tdir == "5":
            self.use_novel = 1
            self.level = 208


        self.seed_list = seed_list
        # Load test based on params
        self.use_seed = False

        self.test = SViz(self.use_mock, self.use_novel, self.level, False, seed, difficulty, use_seed=self.use_seed)

        if int(tdir) < 4 :
            if not check:
                self.test1 = TViz(self.use_mock, self.use_novel, self.level, False, seed, difficulty, use_seed=self.use_seed, task=1)
                self.test2 = TViz(self.use_mock, self.use_novel, self.level, False, seed, difficulty, use_seed=self.use_seed, task=2)
                self.test3 = TViz(self.use_mock, self.use_novel, self.level, False, seed, difficulty, use_seed=self.use_seed, task=3)
                r = np.random.randint(1, 4)
                if r == 1:
                    self.test = self.test1
                elif r == 2:
                    self.test = self.test2
                else:
                    self.test = self.test3
            else:
                self.test = SViz(self.use_mock, self.use_novel, self.level, False, seed, difficulty, use_seed=self.use_seed)
                self.test1 = None
                self.test2 = None
                self.test3 = None

        self.information = self.test.reset()


    def set_seed(self, use_seed):
        self.use_seed = use_seed
        if self.use_seed:
            self.seed = self.seed_list[0]
        self.test = SViz(self.use_mock, self.use_novel, self.level, False, self.seed, self.difficulty, use_seed=self.use_seed)
        if self.tdir < 4:
            self.test1 = TViz(self.use_mock, self.use_novel, self.level, False, self.seed, self.difficulty, use_seed=self.use_seed, task=1)
            self.test2 = TViz(self.use_mock, self.use_novel, self.level, False, self.seed, self.difficulty, use_seed=self.use_seed, task=2)
            self.test3 = TViz(self.use_mock, self.use_novel, self.level, False, self.seed, self.difficulty, use_seed=self.use_seed, task=3)
            np.random.seed(self.seed)
            r = np.random.randint(1, 4)
            if r == 1:
                self.test = self.test1
            elif r == 2:
                self.test = self.test2
            else:
                self.test = self.test3

        self.information = self.test.reset(self.seed)

    def apply_action(self, action):
        action = action['action']
        self.test.step(action)
        self.information = self.test.get_state()
        return self.information['performance']

    def get_feature_vector(self):

        return self.information

    def get_feature_label(self):
        return {'action': self.information['action']}

    def is_episode_done(self):
        return self.test.is_done

    def reset(self, episode):
        if self.tdir < 4:

            if not self.check:
                if self.use_seed:

                    np.random.seed(self.seed_list[episode])
                r = np.random.randint(1, 4)
                if r == 1:
                    self.test = self.test1
                elif r == 2:
                    self.test = self.test2
                else:
                    self.test = self.test3
            else:  # TODO check if this is necessary
               self.test = self.test1

        if self.use_seed:
            self.information = self.test.reset(self.seed_list[episode])
        else:
            self.information = self.test.reset()