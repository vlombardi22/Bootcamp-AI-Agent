"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
from doom_util import v_wrap, set_init, push_and_pull, record_nav, tracker, get_dist, target_in_room, navigate, get_angle
import torch.multiprocessing as mp
from shared import SharedAdam
import matplotlib.pyplot as plt

from Nav2 import Net
import numpy as np
from viz_nav import SailonViz as SViz
import random
import os

os.environ["OMP_NUM_THREADS"] = "5"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.97
MAX_EP = 2000
# HIDDEN_SIZE = 64
# H_SIZE = 32

IS_TEST = False

STATE_SIZE = 12  # 15#13#20#16#20#19#28#24#28#31#24
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
        self.c_list = [{"x_position": 0.0, "y_position": 0.0}, {"x_position": 0.0, "y_position": 448.0},
                       {"x_position": 0.0, "y_position": -448.0}, {"x_position": 448.0, "y_position": 0.0},
                       {"x_position": -448.0, "y_position": 0.0}]

        seed = random.randint(0, 1000)
        use_mock = 0
        use_novel = 0
        level = 0
        difficulty = 'easy'

        self.game = SViz(use_mock, use_novel, level, True, seed, difficulty)

    def break_health(self, items, player):
        temp = []
        if len(items['health']) <= 0:
            return [0.0, 0.0, 0.0]  # , 0.0]

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
            return [0.0, 0.0, 0.0]  # , 0.0]

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
            return [0.0, 0.0, 0.0]  # , 0.0]

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
                  int(player['health']), p_coord]  # padding in case ammo is out

        sensor_vec = avatar + [0.0, 0.0, 0.0, 0.0] + ob_temp

        return np.asarray(sensor_vec), e_count, t_coord, p_coord



    def run(self):  # bookmark

        total_step = 1
        # actions = ['nothing', 'left', 'right', 'backward', 'forward', 'shoot', 'turn_left', 'turn_right']
        actions = ['left', 'right', 'backward', 'forward', 'turn_left', 'turn_right']
        i1 = 5
        i2 = 6
        i3 = 7
        i4 = 8

        while self.g_ep.value < MAX_EP:

            state = self.game.reset()
            items = state['items']

            # target_pillar = items['obstacle'][0]
            player = state['player']
            # obst = self.break_obstacles(items)  # these do not change during an episode
            traps = len(items['trap'])
            _, clip = self.break_ammo(items, player)

            state_vec, _, _, c_coord = self.breaker(state)  # , obst)

            gtype = False

            targe_coord = 1

            if c_coord == 1:
                targ_coord = random.randint(2, 5)

            else:
                targ_coord = 1

            # targ_coord = 3

            # targ_coord = tracker(clip)
            t_x = self.c_list[targ_coord - 1]['x_position']
            t_y = self.c_list[targ_coord - 1]['y_position']

            # prev_dist = get_dist(player,clip)
            prev_dist = get_dist(player, self.c_list[targ_coord - 1])

            # if tracker(clip) == c_coord:
            #    if c_coord == 1:
            #        gtype = False #random.choice([True, False])
            #    else:
            #        gtype = True
            """
            if gtype:
                t_x = float(clip['x_position'])
                t_y = float(clip['y_position'])
                prev_dist = get_dist(player,clip)
                targ_coord = 0
            """

            state_vec[i1] = t_x  # 5
            state_vec[i2] = t_y  # 6
            state_vec[i3] = prev_dist  # 7
            state_vec[i4] = targ_coord

            # t_count = e_count
            kills = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            h = int(player['health'])
            pl_x = player['x_position']
            pl_y = player['y_position']
            ammo = int(player['ammo'])

            goal = False
            count = 0

            initial = True
            while True:
                dud_move = False
                act = self.lnet.choose_action(v_wrap(state_vec[None, :]))
                reward = -1
                new_state, performance, done, victory, dead = self.game.step(actions[act])

                player = new_state['player']

                pl_x2 = player['x_position']
                pl_y2 = player['y_position']
                health = int(player['health'])
                trap = len(new_state['items']['trap'])
                n_ammo = int(player['ammo'])

                nstate_vec, _, _, p_coord, = self.breaker(new_state)  # , obst)
                # c_dist = get_dist(player, clip)
                # c_dist = 0

                # if gtype:
                # c_dist = get_dist(player, clip)
                # else:
                c_dist = get_dist(player, self.c_list[targ_coord - 1])

                """    
                if a_coord == p_coord:
                    t_x = float(clip['x_position'])
                    t_y = float(clip['y_position'])
                    c_dist = get_dist(player, clip)
                    targ_coord = a_coord
                elif p_coord == 1:
                    targ_coord = a_coord
                    t_x = self.c_list[a_coord - 1]['x_position']
                    t_y = self.c_list[a_coord - 1]['y_position']
                    c_dist = get_dist(player, self.c_list[a_coord - 1])
                else:
                    targ_coord = 1
                    t_x = self.c_list[0]['x_position']
                    t_y = self.c_list[0]['y_position']
                    c_dist = get_dist(player, self.c_list[0])
                """

                nstate_vec[i1] = t_x  # 5
                nstate_vec[i2] = t_y  # 6
                nstate_vec[i3] = c_dist  # 7
                nstate_vec[i4] = targ_coord

                if int(pl_x) == int(pl_x2) and int(pl_y) == int(pl_y2):
                    count += 1

                    if act == 0 or act == 1 or act == 2 or act == 3:
                        reward -= 2
                        dud_move = True

                else:
                    count = 0

                if count > 7:
                    reward -= 1
                    count = 0

                if not dud_move:
                    if pl_x2 < 50 and pl_x2 > -50:
                        reward += 0.25
                    elif pl_y2 < 50 and pl_y2 > -50:
                        reward += 0.25

                if c_dist < prev_dist:
                    reward += 1
                """
                if initial and targ_coord == p_coord and targ_coord != c_coord:
                    initial = False
                    reward += 20
                """

                prev_dist = c_dist
                if targ_coord > 0:

                    # if n_ammo > ammo:
                    if c_dist <= 60:
                        # if targ_coord == p_coord:
                        reward += 500
                        done = True
                        goal = True
                """
                else:
                    if n_ammo > ammo:
                        reward += 500
                        done = True
                        goal = True
                """
                pl_x = pl_x2
                pl_y = pl_y2

                if health < h:
                    if trap < traps:
                        traps = trap
                        reward -= 5

                    # else:
                    # reward -= 1
                    h = health

                # ep_r += reward
                ep_r = performance
                buffer_a.append(act)
                buffer_s.append(state_vec)
                buffer_r.append(reward)
                # time.sleep(0.5)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    # if not IS_TEST:
                    push_and_pull(self.opt, self.lnet, self.gnet, done, nstate_vec, buffer_s, buffer_a, buffer_r,
                                  GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information

                        record_nav(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name,
                                   dead, goal, targ_coord, p_coord, gtype)
                        break

                state_vec = nstate_vec
                state = new_state
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(STATE_SIZE, ACTION_SIZE)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    l = input("load Y or N:")
    stric = True
    act_full = {}
    act_net = {}

    if l == "Y":

        f = "track_2.txt"
        act_full = torch.load(f)
        if not stric:
            act_net['pij.weight'] = act_full['pij.weight']
            act_net['pij.bias'] = act_full['pij.bias']
            act_net['vj.weight'] = act_full['vj.weight']
            act_net['vj.bias'] = act_full['vj.bias']
            act_net['pij2.weight'] = act_full['pij2.weight']
            act_net['pij2.bias'] = act_full['pij2.bias']
            act_net['vj2.weight'] = act_full['vj2.weight']
            act_net['vj2.bias'] = act_full['vj2.bias']
            act_net['pi2.weight'] = act_full['pi2.weight']
            act_net['pi2.bias'] = act_full['pi2.bias']
            act_net['v2.weight'] = act_full['v2.weight']
            act_net['v2.bias'] = act_full['v2.bias']
        else:
            act_net = act_full

        gnet.load_state_dict(act_net, strict=stric)

    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    testing = input("test Y or N:")
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
        torch.save(gnet.state_dict(), "seeker_1.txt")

    plt.plot(res)
    plt.ylabel('average')
    plt.xlabel('episode')
    plt.show()