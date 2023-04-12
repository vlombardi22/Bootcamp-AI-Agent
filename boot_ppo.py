"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
import torch
from newutil import v_wrap, record_boot, tracker, get_dist, \
    in_center2, in_center3, tracker2, to_center, to_border, breaker
import torch.multiprocessing as mp
from Nav import Net as nav
from ppo_util import Agent

# import matplotlib.pyplot as plt

import numpy as np
from viz6 import SailonViz as SViz
from viz5 import SailonViz as AViz
from viz7 import SailonViz as HViz
import random
import os
import csv

os.environ["OMP_NUM_THREADS"] = "4"

UPDATE_GLOBAL_ITER = 20
GAMMA = 0.97  # 0.60  # 0.97
MAX_EP = 2000
HIDDEN_SIZE = 32  # 128
H_SIZE = 16  # 64

IS_CONTROL = False
IS_TEST = False

STATE_SIZE = 25
ACTION_SIZE = 4

"""
class Net(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.pi1 = nn.Linear(s_dim, HIDDEN_SIZE)
        self.pij = nn.Linear(HIDDEN_SIZE, H_SIZE)
        self.pij2 = nn.Linear(H_SIZE, H_SIZE)
        self.pi2 = nn.Linear(H_SIZE, a_dim)

        self.v1 = nn.Linear(s_dim, HIDDEN_SIZE)
        self.vj = nn.Linear(HIDDEN_SIZE, H_SIZE)
        self.vj2 = nn.Linear(H_SIZE, H_SIZE)

        self.v2 = nn.Linear(H_SIZE, 1)
        set_init([self.pi1, self.pij, self.pij2, self.pi2, self.v1, self.vj, self.vj2, self.v2])
        # set_init([self.pi1, self.pij, self.pi2, self.v1, self.vj, self.v2])

        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = F.leaky_relu(self.pi1(x))
        v1 = F.leaky_relu(self.v1(x))
        pij = F.leaky_relu(self.pij(pi1))
        vj = F.leaky_relu(self.vj(v1))
        pij2 = F.leaky_relu(self.pij2(pij))
        vj2 = F.leaky_relu(self.vj2(vj))
        logits = self.pi2(pij2)
        values = self.v2(vj2)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss
"""

def helper(player, combat, state):
    check = True
    if not in_center2(player) and combat == 0:
        obst = state['items']['obstacle']
        for o in obst:
            if get_dist(o, player) < 80:
                check = False
    return check


class Worker():

    def __init__(self, gnet, gnav, gammo, global_ep, global_ep_r, res_queue, name, global_kills,
                 global_health, global_ammo, my_queue):

        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet = gnet
        self.g_k = global_kills
        self.g_h = global_health
        self.g_a = global_ammo
        self.gnav = gnav
        self.gammo = gammo
        self.my_q = my_queue



        self.c_list = [{"x_position": 0.0, "y_position": 0.0}, {"x_position": 0.0, "y_position": 458.0},
                       {"x_position": 0.0, "y_position": -458.0}, {"x_position": 458.0, "y_position": 0.0},
                       {"x_position": -458.0, "y_position": 0.0},
                       {"x_position": 0.0, "y_position": 0.0}]
        self.s_list = [{"x_position": 180.0, "y_position": 0.0}, {"x_position": 0.0, "y_position": 180.0},
                       {"x_position": -180.0, "y_position": 0}, {"x_position": 0, "y_position": -180.0}]

        seed = 97  # random.randint(0, 1000)

        self.seed_list = []
        self.use_seed = False

        if IS_TEST:
            self.use_seed = True
        # print(seed)
        if self.use_seed:
            random.seed(seed)
            np.random.seed(seed)

            # for i in range(MAX_EP):
            #    self.seed_list.append(np.random.randint(0, 1000))
            self.seed_list = [np.random.randint(0, 1000) for i in range(MAX_EP)]
        # print(self.seed_list)
        # print(len(self.seed_list))
        # exit()
        use_mock = 0
        use_novel = 0
        level = 0
        difficulty = 'easy'

        self.game = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed)
        self.game2 = AViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed)
        self.game3 = HViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed)

    def helm(self, nav_vec, state, combat, i_targ, clip, med, act, targ_coord, tir, p_coord, player, ammo):

        m_act = np.dtype('int64').type(7)
        n_act = np.dtype('int64').type(7)
        r_act = np.dtype('int64').type(7)
        c_act = np.dtype('int64').type(7)
        override = False

        if act == 0 and ammo > 0:
            if combat == 1:  # and ammo > 0:

                c_act = np.dtype('int64').type(4)

            elif combat == 2:  # and ammo > 0:
                c_act = np.dtype('int64').type(5)

            elif combat == 3:  # and ammo > 0:
                c_act = np.dtype('int64').type(6)
            elif combat == 4 and ammo > 0:
                c_act = np.dtype('int64').type(0)
            elif combat == 5 and ammo > 0:
                c_act = np.dtype('int64').type(1)
        if act == 2 and med:  # len(items['health']) > 0:
            m_coord = tracker(med)
            if m_coord == p_coord:
                if player['angle'] == 90:
                    nav_vec[4] = float(med['x_position'])
                    nav_vec[5] = float(med['y_position'])
                    nav_vec[6] = get_dist(med, player)
                    m_act = self.gammo.choose_action(v_wrap(nav_vec[None, :]))
                else:

                    if 270 > player['angle'] > 90:
                        m_act = np.dtype('int64').type(5)  # 'turn_right'
                    else:
                        m_act = np.dtype('int64').type(4)

            elif p_coord == 1:
                # check = helper(player, combat, state)
                if not in_center3(player) and helper(player, combat, state):  # check:  # doors

                    m_act = to_border(player, m_coord)

                elif player['angle'] != 315:
                    if 315 > player['angle'] > 135:
                        n_act = np.dtype('int64').type(4)  # 'turn_right'
                    else:
                        n_act = np.dtype('int64').type(5)
                else:
                    nav_vec[4] = self.c_list[m_coord - 1]['x_position']
                    nav_vec[5] = self.c_list[m_coord - 1]['y_position']
                    nav_vec[6] = get_dist(player, self.c_list[m_coord - 1])
                    m_act = self.gnav.choose_action(v_wrap(nav_vec[None, :]))

            else:
                m_act = to_center(player, p_coord)

                override = True
        if act == 3 and clip:  # len(items['ammo']) > 0:
            a_coord = tracker(clip)

            if a_coord == p_coord:
                if player['angle'] == 90:
                    nav_vec[4] = float(clip['x_position'])
                    nav_vec[5] = float(clip['y_position'])
                    nav_vec[6] = get_dist(player, clip)
                    r_act = self.gammo.choose_action(v_wrap(nav_vec[None, :]))

                else:
                    if 270 > player['angle'] > 90:
                        r_act = np.dtype('int64').type(5)  # 'turn_right'
                    else:
                        r_act = np.dtype('int64').type(4)
            elif p_coord == 1:
                # check = helper(player, combat, state)

                if not in_center3(player) and helper(player, combat, state):  # check:  # doors
                    r_act = to_border(player, a_coord)

                elif player['angle'] != 315:
                    if 315 > player['angle'] > 135:
                        r_act = np.dtype('int64').type(4)  # 'turn_right'
                    else:
                        r_act = np.dtype('int64').type(5)
                else:
                    nav_vec[4] = self.c_list[a_coord - 1]['x_position']
                    nav_vec[5] = self.c_list[a_coord - 1]['y_position']
                    nav_vec[6] = get_dist(player, self.c_list[a_coord - 1])
                    r_act = self.gnav.choose_action(v_wrap(nav_vec[None, :]))

            else:  # other rooms
                r_act = to_center(player, p_coord)

                override = True

        if act == 1:

            if tir:  # target_in_room(state['enemies'], p_coord):

                if player['angle'] != 90:
                    if 270 > player['angle'] > 90:
                        n_act = np.dtype('int64').type(5)  # 'turn_right'
                    else:
                        n_act = np.dtype('int64').type(4)
                elif p_coord == 1:

                    nav_vec[4] = self.s_list[i_targ]['x_position']
                    nav_vec[5] = self.s_list[i_targ]['y_position']
                    nav_vec[6] = get_dist(player, self.s_list[i_targ])

                    n_act = self.gammo.choose_action(v_wrap(nav_vec[None, :]))

                else:
                    nav_vec[4] = self.c_list[p_coord - 1]['x_position']
                    nav_vec[5] = self.c_list[p_coord - 1]['y_position']
                    nav_vec[6] = get_dist(player, self.c_list[p_coord - 1])
                    n_act = self.gammo.choose_action(v_wrap(nav_vec[None, :]))

            elif p_coord != 1:

                n_act = to_center(player, p_coord)

                override = True
            else:
                # targ_coord = navigate(state['enemies'], p_coord, player)
                # check = helper(player, combat, state)

                if not in_center3(player) and helper(player, combat, state):  # check:  # doors
                    n_act = to_border(player, targ_coord)

                elif player['angle'] != 315:
                    if 315 > player['angle'] > 135:
                        n_act = np.dtype('int64').type(4)  # turn_left
                    else:
                        n_act = np.dtype('int64').type(5)
                else:
                    nav_vec[4] = self.c_list[targ_coord - 1]['x_position']
                    nav_vec[5] = self.c_list[targ_coord - 1]['y_position']
                    nav_vec[6] = get_dist(player, self.c_list[targ_coord - 1])
                    n_act = self.gnav.choose_action(v_wrap(nav_vec[None, :]))

        return r_act, m_act, c_act, n_act, override

    def run(self):  # bookmark

        total_step = 1
        actions = ['left', 'right', 'backward', 'forward', 'shoot', 'turn_left', 'turn_right', 'nothing']
        actions2 = ['left', 'right', 'backward', 'forward', 'turn_left', 'turn_right']
        v_count = 0

        i3 = STATE_SIZE - 1  # 1 # 3
        i4 = STATE_SIZE - 3  # 3 # 2
        i5 = STATE_SIZE - 2  # 2 # 1

        i6 = STATE_SIZE - 4
        task_var = 1.0
        r_list = np.zeros([MAX_EP])  # []

        r_list2 = np.zeros([MAX_EP])  # []
        turn = 0
        switch = 4
        s = 0
        episode = 0

        if self.name == "w01":  # or self.name == "w02":
            task_var = 1.0
            turn = 3  # 6
        if self.name == "w02":
            task_var = 2.0
            turn = 6  # 13
        if self.name == "w03":
            task_var = 3.0
            turn = 9
        my_av = 0.0
        game = None

        while (not IS_TEST and self.g_ep.value < MAX_EP) or (IS_TEST and episode < MAX_EP):
            step = 0

            if not IS_TEST and not IS_CONTROL:

                if turn < switch:
                    task_var = 1.0
                    game = self.game

                else:
                    game = self.game2
                    task_var = 2.0

                    if turn > 7:  # 12: # 12
                        task_var = 3.0
                        game = self.game3
                        if turn > 11:  # 18:
                            turn = 0
                            task_var = 1.0

                            game = self.game
            if IS_TEST:
                s = self.seed_list[episode]

            if IS_TEST or IS_CONTROL:

                if IS_TEST:
                    np.random.seed(s)
                task = np.random.randint(1, 4)
                if task == 1:
                    turn = 0
                    task_var = 1.0

                    game = self.game
                elif task == 2:
                    turn = 4  # 6
                    task_var = 2.0
                    game = self.game2

                elif task == 3:
                    turn = 8  # 3
                    task_var = 3.0
                    game = self.game3
                else:
                    print(task, "!!!!!!!!!!!!!!!!!!!!!!")

            # if IS_TEST:
            #    s = self.seed_list[episode]
            state = game.reset(s)  # self.seed_list[episode])

            # items = state['items']
            player = state['player']
            test_obst = state['items']['obstacle']

            state_vec, nav_vec, e_count, combat, clip, med, targ_coord, tir, seek = breaker(state,
                                                                                            test_obst)  # initial state_vec
            # print(len(state_vec))
            over_ride = False
            hp = int(player['health'])

            t_count = e_count
            kills = 0
            buffer_s, buffer_a, buffer_r = [], [], []

            pl_x = player['x_position']
            pl_y = player['y_position']
            ammo = int(player['ammo'])
            i_targ = 0
            a_count = 0
            s_count = 0
            h_count = 0
            pact = 'nothing'

            stuck = False
            p_coord = tracker(player)

            state_vec[i3] = task_var  # 0.0#1.0

            if e_count > 0:  # can attack
                a_check = seek  # (seeker(state['enemies'], player) and ammo > 0)
                if combat > 0:  # c_act < 7:
                    if a_check:
                        state_vec[i4] = 3.0
                    else:
                        state_vec[i4] = 2.0
                elif a_check:
                    state_vec[i4] = 1.0

            if med and ACTION_SIZE == 4:  # m_act < 7 and ACTION_SIZE == 4:
                if tracker(med) == p_coord and get_dist(med, player) <= 200:
                    state_vec[i6] = 2.0
                else:
                    state_vec[i6] = 1.0

            if clip:  # r_act < 7:
                if tracker(clip) == p_coord and get_dist(clip, player) <= 200:
                    state_vec[i5] = 2.0
                else:
                    state_vec[i5] = 1.0
            rotate_count = 0

            ep_rr = 0.0
            while True:
                step += 1
                reward = -1
                p_coord = tracker(player)
                act, prob, val = self.gnet.choose_action(state_vec)

                #act = self.gnet.choose_action(v_wrap(state_vec[None, :]))
                r_act, m_act, c_act, n_act, override = self.helm(nav_vec, state, combat, i_targ, clip, med, act,
                                                                 targ_coord, tir, p_coord, player, ammo)

                if override:
                    over_ride = True

                elif tracker2(player) != 6:
                    over_ride = False

                if act == 0:
                    my_act = actions[c_act]

                    if c_act == 7:
                        reward = -2
                    else:
                        if c_act == 4 and step < 500:
                            reward += 1
                        reward += 1
                        over_ride = False
                        rotate_count = 0

                else:
                    immobile = False
                    if over_ride and tracker2(player) == 6:
                        my_act = "forward"
                    elif act == 1:
                        my_act = actions2[n_act]

                        # print("nav")
                    elif act == 2:
                        # print("health")
                        if m_act < 7:
                            if turn > 7:  # 12:
                                reward += 0.5
                            my_act = actions2[m_act]

                        else:

                            my_act = 'nothing'
                            immobile = True
                            reward = -2
                    else:
                        # print("ammo")
                        if r_act < 7:
                            my_act = actions2[r_act]
                            if switch <= turn <= 7:  # 12:  # task_var == 2.0:
                                reward += 0.5

                        else:
                            my_act = 'nothing'
                            immobile = True
                            reward = -2
                    if not immobile and stuck:

                        act_temp = pact
                        while act_temp == pact:
                            act_temp = actions2[np.random.randint(0, 4)]

                            my_act = act_temp

                        stuck = False

                if my_act == 'turn_left' or my_act == 'turn_right':
                    rotate_count += 1
                else:
                    rotate_count = 0

                if rotate_count >= 10:
                    stuck = True

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                new_state, performance, done, victory, dead = game.step(my_act)

                player = new_state['player']

                pl_x2 = player['x_position']
                pl_y2 = player['y_position']

                health = int(player['health'])
                items = new_state['items']
                n_ammo = int(player['ammo'])
                nstate_vec, nnav_vec, e_temp, combat, clip, med, targ_coord, tir, seek = breaker(new_state, test_obst)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if get_dist(player, self.s_list[i_targ]) <= 60.0:
                    i_targ += 1
                    if i_targ == 4:
                        i_targ = 0

                if len(items['ammo']) <= 0 and switch <= turn <= 7:  # 12:  # task_var == 2.0:  # and not IS_TEST:
                    done = True
                    victory = True
                if len(items['health']) <= 0 and turn > 7:  # 12:
                    done = True
                    victory = True

                if int(pl_x) == int(pl_x2) and int(pl_y) == int(pl_y2):
                    if my_act == 'left' or my_act == 'right' or my_act == 'backward' or my_act == 'forward':
                        stuck = True
                        pact = my_act

                if victory:
                    v_count += 1
                    if step < 751:  # 1001:
                        reward += 10

                    # if turn < switch:
                    #    reward += 20
                    # else:
                    #    if len(new_state['enemies']) == 0:
                    #        reward += 10

                pl_x = pl_x2
                pl_y = pl_y2

                if e_temp < e_count:
                    reward += 100  # 25#10
                    e_count -= 1
                    kills += 1
                    if step < 100:
                        reward += 20
                    elif step < 400:
                        reward += 10

                if n_ammo > ammo:
                    reward += 10  # 25
                    if act == 3:
                        reward += 50  # 75
                        s_count += 1
                    if task_var == 2.0:
                        reward += 15
                    a_count += 1

                if health > hp:

                    reward += 10
                    if act == 2:
                        reward += 50  # 20
                    if task_var == 3.0:
                        reward += 15
                    h_count += 1

                elif health < hp:
                    reward -= 1

                ammo = n_ammo
                hp = health

                nstate_vec[i3] = task_var  # 0.0#1.0

                if e_count > 0:
                    a_check = seek  # (seeker(new_state['enemies'], player) and ammo > 0)
                    if combat > 0:
                        if a_check:
                            nstate_vec[i4] = 3.0
                        else:
                            nstate_vec[i4] = 2.0
                    elif a_check:
                        nstate_vec[i4] = 1.0

                if med and ACTION_SIZE == 4:
                    if tracker(med) == p_coord and get_dist(med, player) <= 200:
                        nstate_vec[i6] = 2.0
                    else:
                        nstate_vec[i6] = 1.0

                if clip:
                    if tracker(clip) == p_coord and get_dist(clip, player) <= 200:
                        nstate_vec[i5] = 2.0
                    else:
                        nstate_vec[i5] = 1.0
                ep_rr += reward
                if not IS_TEST:
                    buffer_a.append(act)
                    buffer_s.append(state_vec)
                    buffer_r.append(reward)
                    self.gnet.remember(state_vec, act, prob, val, reward, done)
                if (
                        not IS_TEST and total_step % UPDATE_GLOBAL_ITER == 0) or done:  # update global and assign to local net
                    # sync
                    if not IS_TEST:
                        self.gnet.learn()
                    #if len(buffer_s) > 0 and not IS_TEST:
                    #    push_and_pull(self.opt, self.lnet, self.gnet, done, nstate_vec, buffer_s, buffer_a, buffer_r,
                    #                  GAMMA)

                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information

                        if IS_TEST:
                            r_list[episode] = performance  # append(performance)
                            r_list2[episode] = ep_rr  # .append(ep_rr)
                            task = "combat"
                            if task_var == 2:
                                task = "reload"
                            elif task_var == 3:
                                task = "heal"
                            episode += 1

                            # task = get_task(task_var)
                            my_av += performance

                            print(
                                self.name,
                                "Ep:", episode, "enemies:", t_count, "kills:", kills, "victory:", victory,
                                "dead:", dead, "ammo:", a_count,
                                "health:", h_count,
                                "| Ep_r: %.2f" % (my_av / episode), " indiv: %.2f" % performance, task
                            )


                        else:
                            record_boot(self.g_ep, self.g_ep_r, performance, self.res_queue, self.name, t_count, kills,
                                        victory,
                                        dead, a_count, h_count, s_count, task_var, self.g_k, self.g_h, self.g_a, MAX_EP)
                        break
                state_vec = nstate_vec
                state = new_state
                nav_vec = nnav_vec
                total_step += 1
            turn += 1
        if IS_TEST:
            print(v_count)
            av = np.average(r_list)
            print(av)
            self.my_q.put([v_count, np.average(r_list2), av])
        self.res_queue.put(None)


def main(f, my_q, fname, my_r):
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    myshape = np.zeros(STATE_SIZE)
    #print(myshape.shape)

    #exit()
    gnet = Agent(n_actions=ACTION_SIZE,input_dims=myshape.shape, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
    gnav = nav(13, 6)
    gammo = nav(13, 6)

    gnav.load_state_dict(torch.load("ranger_6.txt"))
    gammo.load_state_dict(torch.load("ammo_4.txt"))
    l = "N"  # input("load Y or N:")
    #stric = False
    if IS_TEST:
        l = "Y"
    #    stric = True
    #act_net = {}

    if l == "Y":
        gnet.load_weights(f)
    #    act_net = torch.load(f)
    #    gnet.load_model() #load_state_dict(act_net, strict=stric)

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    global_kills = mp.Value('i', 0)
    global_health = mp.Value('i', 0)
    global_ammo = mp.Value('i', 0)
    if IS_TEST:
        print("testing")
    else:
        print("training")

    # parallel training
    if mp.cpu_count() < 6:
        print("cpu alert")
        exit()
    worker = Worker(gnet, gnav, gammo, global_ep, global_ep_r, res_queue, 0, global_kills, global_health, global_ammo, my_q)
    worker.run()
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    if not IS_TEST:
        print("kills:", global_kills.value)
        print("health:", global_health.value)
        print("ammo:", global_ammo.value)
    m_kills = 250
    m_items = 150

    if not IS_TEST and (IS_CONTROL or (
            global_kills.value > m_kills and global_health.value > m_items and global_ammo.value > m_items)):
        #torch.save(gnet.state_dict(), f)
        gnet.save_weights(f)
        # plt.plot(res)
        # print(res)
        my_r2 = np.add(my_r, res)
        # plt.ylabel('average')
        # plt.xlabel('episode')
        # plt.show()
        # plt.savefig(fname)
        # plt.clf()
        return True, my_r2

    if IS_TEST:
        return True, my_r
    return False, my_r


if __name__ == "__main__":
    x = 0
    rang = 30
    test_ep = 1000

    control = "N"  # input("control Y or N:")
    testing = "N"  # input("test Y or N:")
    is_load = "N"  # input("continue Y or N:")

    if control == "Y" or control == "y":
        IS_CONTROL = True
    if testing == "Y" or testing == "y":
        IS_TEST = True

    # MAX_EP = 10
    if IS_TEST:
        MAX_EP = test_ep

    my_q = mp.Queue()
    my_r = np.zeros([MAX_EP])
    pref = "ppoboot_"
    if IS_CONTROL:
        pref = "ppocontrol_"

    for ind in range(rang):
        n = ind + x
        f0 = pref + str(n)
        f1 = f0 + ".txt"
        fname = f0 + ".png"
        if IS_TEST and not os.path.exists(f1):
            print("file:", f1, "does not exist")
            break
        print(f1)
        while True:
            temp, my_r = main(f1, my_q, fname, my_r)
            if temp:

                break
            else:
                print("retraining")

    IS_TEST = True

    if IS_TEST:
        MAX_EP = test_ep
    my_q = mp.Queue()

    for ind in range(rang):
        n = ind + x
        f0 = pref + str(n)
        f1 = f0 + ".txt"
        fname = f0 + ".png"
        f3 = "act_" + f1
        if IS_TEST and not os.path.exists(f3):
            print("file:", f1, "does not exist")
            break
        print(f1)
        temp, _ = main(f1, my_q, fname, my_r)
    # name of csv file
    filename = "boot_ppo.csv"

    if IS_CONTROL:
        filename = "control_ppo.csv"

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
                    my_r = np.add(my_r, l)

                header = False

    # writing to csv file

    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        head = np.zeros([len(my_r)])

        head[0] = rang
        rows = [head, my_r]

        csvwriter.writerows(rows)
    csvfile.close()

    my_q.put(None)

    f = open("myout_task123_ppo.txt", "w")
    if IS_CONTROL:
        f.write("control\n")
    else:
        f.write("boot\n")
    while True:
        r = my_q.get()
        if r is not None:
            print(r)
            mystr = str(r) + "\n"
            f.write(mystr)

        else:
            break

    f.close()
    print("done")