#!/usr/bin/env python3
# import json
# import random

from viz_task45 import SailonViz as SViz

import numpy as np


class TestHandler:

    # Init function, accepts connection and address info
    def __init__(self, seed_list, domain: str = 'cartpole', novelty: int = 0, difficulty: str = 'easy',
                 seed: int = 123, trial_novelty: int = 0, day_offset: int = 0, use_img: bool = False,
                 path: str = "env_generator/envs/", use_gui: bool = False, check: bool = False):

        # Set parameters
        self.seed = seed
        self.domain = domain
        self.novelty = novelty
        self.difficulty = 'easy'#difficulty
        self.check = check
        self.trial_novelty = trial_novelty
        #self.difficulty = difficulty
        self.day_offset = day_offset
        self.use_img = use_img
        self.path = path
        self.use_gui = use_gui
        self.use_mock = 0
        self.use_novel = 1
        self.level = 208


        self.seed_list = seed_list
        #difficulty = 'easy'
        # Load test based on params
        self.use_seed = False
        self.test = SViz(self.use_mock, self.use_novel, self.level, False, seed, difficulty, use_seed=self.use_seed)
        """
        if not check:
            self.test1 = SViz(self.use_mock, self.use_novel, self.level, False, seed, difficulty, use_seed=self.use_seed)
            self.test2 = AViz(self.use_mock, self.use_novel, self.level, False, seed, difficulty, use_seed=self.use_seed)#False)
            self.test3 = HViz(self.use_mock, self.use_novel, self.level, False, seed, difficulty, use_seed=self.use_seed)
            # self.test.reset()
            # Get first information
            r = np.random.randint(1, 4)
            if r == 1:
                self.test = self.test1  # self.test.get_state(initial=True)
            elif r == 2:
                self.test = self.test2
            else:
                self.test = self.test3
        else:
            self.test = SViz(self.use_mock, self.use_novel, self.level, False, seed, difficulty, use_seed=self.use_seed)
            self.test1 = None
            self.test2 = None
            self.test3 = None
        """
        self.information = self.test.reset()


    def set_seed(self, use_seed):
        self.use_seed = use_seed
        if self.use_seed:
            self.seed = self.seed_list[0]
        self.test = SViz(self.use_mock, self.use_novel, self.level, False, self.seed, self.difficulty, use_seed=self.use_seed)
        """
        self.test1 = SViz(self.use_mock, self.use_novel, self.level, False, self.seed, self.difficulty, use_seed=self.use_seed)
        self.test2 = AViz(self.use_mock, self.use_novel, self.level, False, self.seed, self.difficulty, use_seed=self.use_seed)
        self.test3 = HViz(self.use_mock, self.use_novel, self.level, False, self.seed, self.difficulty, use_seed=self.use_seed)
        np.random.seed(self.seed)
        r = np.random.randint(1, 4)
        if r == 1:
            self.test = self.test1  # self.test.get_state(initial=True)
        elif r == 2:
            self.test = self.test2
        else:
            self.test = self.test3
        """
        self.information = self.test.reset(self.seed)

    def apply_action(self, action):
        action = action['action']
        self.test.step(action)  # act(action)
        self.information = self.test.get_state()
        return self.information['performance']

    def get_feature_vector(self):

        return self.information  # ['sensors']

    def get_feature_label(self):
        return {'action': self.information['action']}

    def is_episode_done(self):
        return self.test.is_done

    def reset(self, episode):
        """
        if not self.check:
            if self.use_seed:
                #self.seed_list[episode]
                np.random.seed(self.seed_list[episode])
            r = np.random.randint(1, 4)
            if r == 1:
                self.test = self.test1  # self.test.get_state(initial=True)
            elif r == 2:
                self.test = self.test2
            else:
                self.test = self.test3
        # else:
        #    self.test = self.test1
        """
        if self.use_seed:
            self.information = self.test.reset(self.seed_list[episode])
        else:
            self.information = self.test.reset()