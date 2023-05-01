"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
from boot_utils.myutil import v_wrap, push_and_pull, record_nav, tracker, get_dist, target_in_room, navigate
import torch.multiprocessing as mp
from boot_utils.shared import SharedAdam
import matplotlib.pyplot as plt

from boot_utils.Nav import Net
import numpy as np
from viz_nav import SailonViz as SViz
import random
import os

os.environ["OMP_NUM_THREADS"] = "5"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.97
MAX_EP = 2000


IS_TEST = False

STATE_SIZE = 20
ACTION_SIZE = 6


# 36.6422 distance between obstacle and player for collision

class Worker(mp.Process):

    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, f, l, stric):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(STATE_SIZE, ACTION_SIZE)  # local network
        self.r_side = np.pi / 64
        if l == "Y":
            self.lnet.load_state_dict(f, strict=stric)
        self.c_list = [{"x_position": 0.0, "y_position": 0.0}, {"x_position": 0.0, "y_position": 488.0},
                       {"x_position": 0.0, "y_position": -488.0}, {"x_position": 488.0, "y_position": 0.0},
                       {"x_position": -488.0, "y_position": 0.0}]

        seed = random.randint(0, 1000)
        use_mock = 0
        use_novel = 0
        level = 0
        difficulty = 'easy'

        self.game = SViz(use_mock, use_novel, level, True, seed, difficulty)

    def break_health(self, items, player):
        temp = []
        if len(items['health']) <= 0:
            return [0.0, 0.0, 0.0]

        min_dist = 10000
        m_health = items['health'][0]
        for h in items['health']:
            dist = get_dist(player, h)
            if min_dist > dist:
                min_dist = dist
                m_health = h
        temp.append(float(m_health['x_position']))
        temp.append(float(m_health['y_position']))
        temp.append(min_dist)

        return temp

    def break_ammo(self, items, player):
        temp = []
        if len(items['ammo']) <= 0:
            return [0.0, 0.0, 0.0], None  # , 0.0]

        min_dist = 10000
        m_ammo = items['ammo'][0]
        for a in items['ammo']:
            dist = get_dist(player, a)
            if min_dist > dist:
                min_dist = dist
                m_ammo = a
        temp.append(float(m_ammo['x_position']))
        temp.append(float(m_ammo['y_position']))
        temp.append(min_dist)

        return temp, m_ammo

    def break_traps(self, items, player):
        temp = []
        if len(items['trap']) <= 0:
            return [0.0, 0.0, 0.0]

        min_dist = 10000
        m_trap = items['trap'][0]
        for t in items['trap']:
            dist = get_dist(player, t)
            if min_dist > dist:
                min_dist = dist
                m_trap = t

        temp.append(float(m_trap['x_position']))
        temp.append(float(m_trap['y_position']))
        temp.append(min_dist)
        return temp

    def break_obstacles(self, items, player):
        temp = []
        if len(items['obstacle']) <= 0:
            return [0.0, 0.0, 0.0]

        min_dist = 10000
        m_obst = items['obstacle'][0]
        for o in items['obstacle']:
            dist = get_dist(player, o)
            if min_dist > dist:
                min_dist = dist
                m_obst = o

        temp.append(float(m_obst['x_position']))
        temp.append(float(m_obst['y_position']))
        temp.append(min_dist)
        return temp

    def breaker(self, state):  # , obstacles):
        enemies = state['enemies']
        items = state['items']
        player = state['player']
        t_coord = 0
        p_coord = tracker(player)
        if not target_in_room(enemies, p_coord):
            t_coord = navigate(enemies, p_coord, player)

        e_count = len(enemies)
        ob_temp = self.break_obstacles(items, player)

        t_temp = self.break_traps(items, player)

        avatar = [float(player['x_position']), float(player['y_position']), float(player['angle']),
                  int(player['health']), 20]  # padding in case ammo is out

        sensor_vec = avatar + [0.0, 0.0, 0.0] + ob_temp + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + t_temp

        return np.asarray(sensor_vec), e_count, t_coord, p_coord


    def run(self):

        total_step = 1
        actions = ['left', 'right', 'backward', 'forward', 'turn_left', 'turn_right']

        while self.g_ep.value < MAX_EP:
            step = 0
            state = self.game.reset()

            items = state['items']

            player = state['player']
            traps = len(items['trap'])
            _, clip = self.break_ammo(items, player)

            state_vec, e_count, targ_coord, c_coord = self.breaker(state)  # , obst)

            gtype = False
            test = 1
            if c_coord == 1:
                test = random.randint(0, 5)

            if c_coord == tracker(clip) and test == 1:
                targ_coord = c_coord

                t_x = clip['x_position']
                t_y = clip['y_position']

                prev_dist = get_dist(player, clip)


                state_vec[5] = t_x
                state_vec[6] = t_y
                state_vec[7] = prev_dist


                buffer_s, buffer_a, buffer_r = [], [], []
                ep_r = 0.
                h = int(player['health'])
                pl_x = player['x_position']
                pl_y = player['y_position']
                ammo = int(player['ammo'])

                goal = False
                count = 0

                while True:
                    act = self.lnet.choose_action(v_wrap(state_vec[None, :]))
                    reward = -1
                    new_state, _, done, victory, dead = self.game.step(actions[act])

                    step += 1
                    player = new_state['player']

                    pl_x2 = player['x_position']
                    pl_y2 = player['y_position']
                    health = int(player['health'])
                    trap = len(new_state['items']['trap'])
                    n_ammo = int(player['ammo'])

                    nstate_vec, e_temp, t_coord, p_coord, = self.breaker(new_state)  # , obst)
                    c_dist = 0

                    if clip:
                        c_dist = get_dist(player, clip)
                    nstate_vec[5] = t_x  # 5
                    nstate_vec[6] = t_y  # 6
                    nstate_vec[7] = c_dist  # 7

                    if int(pl_x) == int(pl_x2) and int(pl_y) == int(pl_y2):
                        count += 1

                        if act == 0 or act == 1 or act == 2 or act == 3:
                            reward -= 2

                    else:
                        count = 0

                    if count > 7:
                        count = 0


                    if c_dist < prev_dist:
                        reward += 1

                    prev_dist = c_dist

                    if n_ammo > ammo:

                        reward += 500
                        done = True
                        goal = True

                    pl_x = pl_x2
                    pl_y = pl_y2

                    if health < h:
                        if trap < traps:
                            traps = trap
                            reward -= 5
                        h = health

                    ep_r += reward
                    buffer_a.append(act)
                    buffer_s.append(state_vec)
                    buffer_r.append(reward)
                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                        # sync
                        push_and_pull(self.opt, self.lnet, self.gnet, done, nstate_vec, buffer_s, buffer_a, buffer_r,
                                      GAMMA)
                        buffer_s, buffer_a, buffer_r = [], [], []

                        if done:  # done and print information

                            record_nav(self.g_ep, self.g_ep_r, c_dist, self.res_queue, self.name,
                                       dead, goal, targ_coord, p_coord, gtype)
                            break

                    state_vec = nstate_vec
                    total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(STATE_SIZE, ACTION_SIZE)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    l = "N"#input("load Y or N:")
    stric = True
    act_full = {}
    act_net = {}

    if l == "Y":

        f = "ammo_1.txt"
        act_net = torch.load(f)


        gnet.load_state_dict(act_net, strict=stric)

    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    testing = "N"#input("test Y or N:")
    cp_count = 1

    if testing == "Y":
        print("Testing")
        MAX_EP = 1000
        IS_TEST = True
    else:
        print("Training")
        cp_count = 5

    # parallel training
    if mp.cpu_count() < 6:
        print("cpu alert")
        exit()
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, act_net, l, stric) for i in
               range(cp_count)]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    if not IS_TEST:
        torch.save(gnet.state_dict(), "ammo_2.txt")

    plt.plot(res)
    plt.ylabel('average')
    plt.xlabel('episode')
    plt.show()
