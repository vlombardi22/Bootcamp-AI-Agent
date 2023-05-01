
import numpy as np

import random
from boot_utils.test_handler import TestHandler
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

TRAIN_EPS = 1000  #2000
TEST_EPS = 1000
IS_TEST = False


# Wrapper class for keras-rl dqn learning
class CWrapper:

    def __init__(self, novelty, difficulty, my_res, my_jump, my_asym, my_info, seed=97):
        # Parameters
        self.seed = seed
        self.my_res = my_res
        self.metrics = [0.0, 0.0, 0.0]
        self.kills = 0
        check = False
        # Internal vars
        self.env = None
        self.my_pref = 0

        self.my_jump = my_jump
        self.my_asym = my_asym
        self.my_info = my_info
        self.ep_count = 0
        self.ep_reward = 0
        self.step_count = 0
        self.last_health = None
        self.walls = None
        self.a_count = 0
        self.h_count = 0
        self.e_count = 0

        random.seed(seed)
        np.random.seed(seed)

        self.seed_list = [np.random.randint(0, 1000) for i in range(TEST_EPS)]

        self.env = TestHandler(self.seed_list, domain='vizdoom', novelty=novelty, trial_novelty=novelty,
                               difficulty=difficulty, seed=self.seed, check=check, tdir=1)

        self.action_space = range(8)
        self.observation_space = np.zeros(28)

    def wipe(self, my_info, testing):
        self.my_info = my_info
        if testing:
            self.metrics = [0.0, 0.0, 0.0]

    def analyze(self):
        self.metrics[1] = float(self.metrics[1] / TEST_EPS)
        self.metrics[2] = float(self.metrics[2] / TEST_EPS)
        return self.metrics

    def transform(self, state):
        # Add player
        obs_state = [round((state['player']['x_position'] + 512) / 1024, 2),
                     round((state['player']['y_position'] + 512) / 1024, 2), round(state['player']['angle'] / 360, 2),
                     round(state['player']['health']), round(state['player']['ammo'])]

        # Add enemies
        for i in range(3):
            if i < len(state['enemies']):
                obs_state.append(round((state['enemies'][i]['x_position'] + 512) / 1024, 2))
                obs_state.append(round((state['enemies'][i]['y_position'] + 512) / 1024, 2))
                obs_state.append(round((state['enemies'][i]['health'] + 512) / 1024, 2))
            else:
                obs_state.append(0.0)
                obs_state.append(0.0)
                obs_state.append(0.0)
        for i in range(3):
            if i < len(state['items']['health']):
                obs_state.append(round((state['items']['health'][i]['x_position'] + 512) / 1024, 2))
                obs_state.append(round((state['items']['health'][i]['y_position'] + 512) / 1024, 2))
            else:
                obs_state.append(0.0)
                obs_state.append(0.0)
        for i in range(3):
            if i < len(state['items']['ammo']):
                obs_state.append(round((state['items']['ammo'][i]['x_position'] + 512) / 1024, 2))
                obs_state.append(round((state['items']['ammo'][i]['y_position'] + 512) / 1024, 2))
            else:
                obs_state.append(0.0)
                obs_state.append(0.0)

        obs_state.append(self.check_shoot(state))
        obs_state.append(self.env.test.get_task())

        return obs_state

    def set_seed(self, use_seed):
        self.ep_count = 0
        self.env.set_seed(use_seed)

    def step(self, action):

        obs, pref, done, victory, dead = self.env.test.step(self.action_trans(action))
        task = self.env.test.get_task()

        # Health
        health = 0
        for enemy in obs['enemies']:
            health = health + enemy['health']

        h_count = len(obs['items']['health'])
        a_count = len(obs['items']['ammo'])
        e_count = len(obs['enemies'])
        if dead:
            reward = -300
        elif victory:
            reward = 500

        else:
            reward = -1

        if self.last_health is not None:
            if health < self.last_health:
                reward += 25

        if e_count < self.e_count:
            reward += 200
            self.kills += 1
            if task == 1:
                reward += 50
        if h_count < self.h_count:
            reward += 200
            if task == 3:
                reward += 50
        if a_count < self.a_count:
            reward += 200
            if task == 2:
                reward += 50

        self.ep_reward += reward
        if done:
            if not IS_TEST:
                p = pref
                if self.step_count > 0:
                    p = self.my_pref * 0.99 + pref * 0.01
                self.my_pref = p

                if self.ep_count >= len(self.my_res):
                    self.my_res.append(p)
                else:
                    self.my_res[self.ep_count] += p

                if self.ep_count < 200:
                    self.my_jump.append(pref)
                if self.ep_count >= 800:
                    self.my_asym.append(pref)

            else:
                if victory:
                    self.metrics[0] += 1
                self.metrics[1] += reward
                self.metrics[2] += pref

            self.my_info.append([self.ep_count, self.ep_reward, self.step_count, pref, self.kills, a_count, h_count])
        self.e_count = e_count
        self.a_count = a_count
        self.h_count = h_count

        self.last_health = health
        self.step_count += 1

        return self.transform(obs), reward, done, {}

    def reset(self):
        self.last_health = None
        self.step_count = 0

        self.ep_reward = 0
        self.env.reset(self.ep_count)
        self.ep_count += 1

        ob = self.env.get_feature_vector()
        self.kills = 0
        self.h_count = len(ob['items']['health'])
        self.a_count = len(ob['items']['ammo'])
        self.e_count = len(ob['enemies'])

        self.walls = ob['walls']
        return self.transform(ob)

    def action_trans(self, action):
        action_name = ""
        if action == 0:
            action_name = 'left'
        elif action == 1:
            action_name = 'right'
        elif action == 2:
            action_name = 'backward'
        elif action == 3:
            action_name = 'forward'
        elif action == 4:
            action_name = 'turn_left'
        elif action == 5:
            action_name = 'turn_right'
        elif action == 6:
            action_name = 'shoot'
        return action_name

    # player shoot enemy
    def check_shoot(self, state):
        shoot = False
        for ind, val in enumerate(state['enemies']):
            angle, sign = self.get_angle(val, state['player'])
            if angle < np.pi / 8:
                for wall in self.walls:
                    if self.intersect({'x': state['player']['x_position'], 'y': state['player']['y_position']},
                                      {'x': val['x_position'], 'y': val['y_position']},
                                      {'x': wall['x1'], 'y': wall['y1']},
                                      {'x': wall['x2'], 'y': wall['y2']}):
                        return shoot
                shoot = True

        return shoot

    # Utility function for getting angle from B-direction to A
    def get_angle(self, player, enemy):
        pl_x = player['x_position']
        pl_y = player['y_position']

        en_x = enemy['x_position']
        en_y = enemy['y_position']
        en_ori = enemy['angle'] * 2 * np.pi / 360

        # Get angle between player and enemy
        # Convert enemy ori to unit vector
        v1_x = np.cos(en_ori)
        v1_y = np.sin(en_ori)

        enemy_vector = np.asarray([v1_x, v1_y]) / np.linalg.norm(np.asarray([v1_x, v1_y]))

        # If its buggy throw random value out
        if np.linalg.norm(np.asarray([pl_x - en_x, pl_y - en_y])) == 0:
            return np.random.rand() * 3.14

        enemy_face_vector = np.asarray([pl_x - en_x, pl_y - en_y]) / np.linalg.norm(
            np.asarray([pl_x - en_x, pl_y - en_y]))

        angle = np.arccos(np.clip(np.dot(enemy_vector, enemy_face_vector), -1.0, 1.0))

        sign = np.sign(np.linalg.det(
            np.stack((enemy_vector[-2:], enemy_face_vector[-2:]))
        ))

        return angle, sign

    def ccw(self, A, B, C):
        return (C['y'] - A['y']) * (B['x'] - A['x']) > (B['y'] - A['y']) * (C['x'] - A['x'])

    # Return true if line segments AB and CD intersect
    def intersect(self, A, B, C, D):
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)


if __name__ == "__main__":
    train = True
    IS_TEST = False
    my_q = []

    jump = []
    asympt = []
    my_info = []

    train_metrics = []

    my_results = []

    rang = 10
    is_load = "N"
    env = CWrapper(200, 'easy', my_q, jump, asympt, my_info, seed=97)
    nb_actions = len(env.action_space) - 1
    step_length = 2000

    targ_steps = step_length * TRAIN_EPS

    for i in range(rang):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics! 50000
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                       target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

        # Ctrl + C. 150000
        # 3000000
        my_str = "vizdoom_task123" + str(i) + ".h5f"

        dqn.load_weights(my_str)
        dqn.fit(env, nb_steps=targ_steps, visualize=False, verbose=2, nb_max_episode_steps=step_length)
        my_str2 = "vizdoom_task5" + str(i) + ".h5f"
        f3 = "vizdoom_task5" + str(i) + "raw.csv"
        with open(f3, 'w', newline='') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(my_info)
        csvfile.close()

        # After training is done, we save the final weights.
        dqn.save_weights(my_str2, overwrite=True)
        m_asym = np.average(asympt)
        m_jump = np.average(jump)
        train_metrics.append([m_asym, m_jump])
        env.ep_count = 0
        my_info = []
        env.wipe(my_info, False)
        # Finally, evaluate our algorithm for 5 episodes.

    IS_TEST = True
    for i in range(rang):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics! 50000
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        env.set_seed(True)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                       target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
        my_str = "vizdoom_task5" + str(i) + ".h5f"
        f3 = "vizdoom_task5" + str(i) + "rawtest.csv"
        dqn.load_weights(my_str)

        t = dqn.test(env, nb_episodes=TEST_EPS, visualize=False)
        with open(f3, 'w', newline='') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(my_info)
        csvfile.close()
        my_results.append(env.analyze())
        my_info = []
        env.wipe(my_info, True)

    filename = "dqn_task5.csv"

    if is_load == "Y" or is_load == "y":
        with open(filename, 'r') as file:
            csvFile = csv.reader(file)
            header = True
            # displaying the contents of the CSV file

            for lines in csvFile:
                if lines and header:
                    h = np.asarray(lines, dtype="float64")
                    rang += h[0]
                if lines and not header:
                    l = np.asarray(lines, dtype="float64")
                    my_q = np.add(my_q, l)

                header = False

    # writing to csv file

    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        head = np.zeros([len(my_q)])

        head[0] = rang
        rows = [head, my_q]

        csvwriter.writerows(rows)
    csvfile.close()
    print(len(my_q))
    f = open("myout_dqn_task5.txt", "w")
    f.write("dqn\n")
    for r in my_results:
        mystr = str(r) + "\n"
        f.write(mystr)

    f.write("other\n")
    print("other")
    for i in train_metrics:
        mystr2 = str(i) + "\n"
        f.write(mystr2)
        print(i)
    f.close()
    print("done")
