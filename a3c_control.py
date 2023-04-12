"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
import torch
import torch.nn as nn
from newutil import v_wrap, set_init, push_and_pull, record_base, tracker, get_dist, break_ammo, \
    break_health, get_angle, h_check, gunner
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared import SharedAdam

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
HIDDEN_SIZE = 64  # 32  # 128
H_SIZE = 32  # 16  # 64

IS_CONTROL = False
IS_TEST = False

STATE_SIZE = 28  # 170#26  # 19#15#13  # 12#13  # 15#14#11#8#10#12  #12#8#9#11
ACTION_SIZE = 7  # 8


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


def break_obstacles(items, player):
    # nav_obst = []
    min_dist = 10000
    m_obst = None

    for o in items['obstacle']:
        dist = get_dist(player, o)

        # if target_sighted(o, player):
        #    ob_list.append(o)

        if min_dist > dist:
            min_dist = dist
            m_obst = o

    if not m_obst:  # len(items['obstacle']) <= 0:
        strat_obst = [0.0, 0.0, -1.0, 0.0]

    else:
        angle, _ = get_angle(m_obst, player, 0.0)
        angle = angle * 180 / np.pi
        strat_obst = [float(m_obst['x_position']), float(m_obst['y_position']), min_dist, angle]

    return strat_obst


def break_enemy(enemies, player):
    # nav_enemy = []
    # strat_enemy = []
    min_dist = 10000
    m_enemy = None
    elist = {}
    for e in enemies:  # bookmark e
        dist = get_dist(player, e)
        elist[int(e['id'])] = h_check(e)
        if min_dist > dist:  # and target_sighted(e, player):
            min_dist = dist
            m_enemy = e

    if not m_enemy:
        strat_enemy = [0.0, 0.0, 0.0, -1.0, 0.0]
    else:
        angle, _ = get_angle(m_enemy, player, 0.0)
        angle = angle * 180 / np.pi
        strat_enemy = [m_enemy['x_position'], m_enemy['y_position'], h_check(m_enemy), get_dist(m_enemy, player),
                       angle]

    return strat_enemy, elist


def breaker(state, wall):  # bookmark
    enemies = state['enemies']
    items = state['items']
    player = state['player']

    strat_enemy, e_list = break_enemy(enemies, player)
    p_coord = tracker(player)
    strat_obst = break_obstacles(items, player)

    e_count = len(enemies)
    a_count = len(items['ammo'])
    h_count = len(items['health'])
    # nav_trap = break_traps(items, player)
    # c_act = 0
    # for e in enemies:
    #    if gunner(e, player, 0.0, 18):
    #        c_act = 1

    _, strat_ammo = break_ammo(items, player, p_coord, enemies)
    _, strat_health = break_health(items, player, p_coord, enemies)
    sensor_vec = [float(player['x_position']), float(player['y_position']), float(player['angle']), int(player['ammo']),
                  int(player['health']), e_count] + strat_enemy + [a_count] + strat_ammo + [
                     h_count] + strat_health + strat_obst + [0.0, 0, 0]  # wall +[0.0]

    # print(len(sensor_vec))
    # exit()

    return np.asarray(sensor_vec), e_count, e_list


class Worker(mp.Process):

    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, f, l, stric, my_queue):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(STATE_SIZE, ACTION_SIZE)
        self.my_q = my_queue

        if l == "Y":
            self.lnet.load_state_dict(f, strict=stric)

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

        use_mock = 0
        use_novel = 0
        level = 0
        difficulty = 'easy'

        self.game = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed)
        self.game2 = AViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed)
        self.game3 = HViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed)

    # player shoot enemy
    def check_shoot(self, state, walls):
        shoot = False
        for ind, val in enumerate(state['enemies']):
            angle, sign = self.get_angle(val, state['player'])
            if angle < np.pi / 8:
                for wall in walls:
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

    def run(self):  # bookmark
        wall = [-320.0, -64.0, 320.0, 320.0, 320.0, 320.0, 320.0, 64.0, 320.0, 64.0, -320.0, -320.0, -320.0, -320.0,
                -320.0, -64.0, 64.0, 64.0, 384.0, 320.0, 64.0, 320.0, 320.0, 320.0, -64.0, -64.0, 320.0, 384.0, -64.0,
                -320.0, 384.0, 384.0, -320.0, -320.0, 384.0, 512.0, -320.0, 320.0, 512.0, 512.0, 320.0, 320.0, 512.0,
                384.0, 320.0, 64.0, 384.0, 384.0, -384.0, -384.0, 320.0, 64.0, 320.0, 320.0, -64.0, -320.0, 320.0,
                384.0, 64.0, 64.0, 384.0, 384.0, 64.0, 320.0, 384.0, 512.0, 320.0, 320.0, 512.0, 512.0, 320.0, -320.0,
                512.0, 384.0, -320.0, -320.0, 384.0, 384.0, -320.0, -64.0, 384.0, 320.0, -64.0, -64.0, -512.0, -384.0,
                320.0, 320.0, -64.0, -320.0, -320.0, -320.0, 64.0, 64.0, -320.0, -384.0, 64.0, 320.0, -384.0, -384.0,
                320.0, 320.0, -384.0, -512.0, 320.0, -320.0, -512.0, -512.0, -320.0, -320.0, -512.0, -384.0, -320.0,
                -64.0, -384.0, -384.0, -64.0, -64.0, -384.0, -320.0, -384.0, -320.0, 64.0, 64.0, -320.0, -320.0, 64.0,
                320.0, -320.0, -384.0, -64.0, -64.0, -384.0, -384.0, -64.0, -320.0, -384.0, -512.0, -320.0, -320.0,
                -512.0, -512.0, -320.0, 320.0]
        total_step = 1
        actions = ['left', 'right', 'backward', 'forward', 'shoot', 'turn_left', 'turn_right', 'nothing']
        v_count = 0

        i3 = STATE_SIZE - 1  # 1 # 3
        i4 = STATE_SIZE - 2
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

            if IS_TEST:
                s = self.seed_list[episode]
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

            """
            print(type(state['walls'][0]))
            temp = []
            for t in state['walls']:
                for t2 in t.values():
                    temp.append(t2)
            print(temp)
            print("!!!!!!!!!!!!!!!!!!!!!!!!")
            print(len(temp))

            #print(state['walls']#.items())
            exit()
            """

            walls = state['walls']

            state_vec, e_count, e_list = breaker(state, wall)  # initial state_vec

            hp = int(player['health'])

            t_count = e_count
            kills = 0
            buffer_s, buffer_a, buffer_r = [], [], []

            pl_x = player['x_position']
            pl_y = player['y_position']
            ammo = int(player['ammo'])
            a_count = 0
            h_count = 0

            state_vec[i3] = task_var  # 0.0#1.0
            state_vec[i4] = self.check_shoot(state, walls)
            ep_rr = 0.0
            while True:
                step += 1
                reward = -1
                fired = False
                # state_vec[i4] = self.check_shoot(state)
                act = self.lnet.choose_action(v_wrap(state_vec[None, :]))
                my_act = actions[act]
                if my_act == "shoot":
                    if ammo > 0:
                        fired = True
                    # else:
                    #    reward -= 1

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                new_state, performance, done, victory, dead = game.step(my_act)

                player = new_state['player']

                pl_x2 = player['x_position']
                pl_y2 = player['y_position']

                health = int(player['health'])
                items = new_state['items']
                n_ammo = int(player['ammo'])
                nstate_vec, e_temp, elist = breaker(new_state, wall)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                if len(items['ammo']) <= 0 and switch <= turn <= 7:  # 12:  # task_var == 2.0:  # and not IS_TEST:
                    done = True
                    victory = True
                if len(items['health']) <= 0 and turn > 7:  # 12:
                    done = True
                    victory = True

                if int(pl_x) == int(pl_x2) and int(pl_y) == int(pl_y2):
                    if my_act == 'left' or my_act == 'right' or my_act == 'backward' or my_act == 'forward' or my_act == 'nothing':
                        reward -= 1

                for key in elist.keys():

                    if elist[key] < e_list[key]:

                        if fired:
                            reward += 40  # 10
                e_list = elist

                if victory:
                    v_count += 1
                    reward += 200
                    if step < 751:  # 1001:
                        reward += 10

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
                    reward += 75  # 60  # 25
                    if task_var == 2.0:
                        reward += 15
                    a_count += 1

                if health > hp:

                    reward += 75  # 60
                    if task_var == 3.0:
                        reward += 15
                    h_count += 1

                elif health < hp:
                    reward -= 1

                ammo = n_ammo
                hp = health

                nstate_vec[i3] = task_var  # 0.0#1.0
                nstate_vec[i4] = self.check_shoot(new_state, walls)
                ep_rr += reward
                if not IS_TEST:
                    buffer_a.append(act)
                    buffer_s.append(state_vec)
                    buffer_r.append(reward)

                if (
                        not IS_TEST and total_step % UPDATE_GLOBAL_ITER == 0) or done:  # update global and assign to local net
                    # sync

                    if len(buffer_s) > 0 and not IS_TEST:
                        push_and_pull(self.opt, self.lnet, self.gnet, done, nstate_vec, buffer_s, buffer_a, buffer_r,
                                      GAMMA)

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
                            record_base(self.g_ep, self.g_ep_r, performance, self.res_queue, self.name, t_count, kills,
                                        victory,
                                        dead, a_count, h_count, task_var)
                        break
                state_vec = nstate_vec
                state = new_state
                total_step += 1
            turn += 1
        if IS_TEST:
            print(v_count)
            av = np.average(r_list)
            print(av)
            self.my_q.put([v_count, np.average(r_list2), av])
        self.res_queue.put(None)


def main(f, my_q, fname, my_r):
    gnet = Net(STATE_SIZE, ACTION_SIZE)  # global network    global_kills = mp.Value('i', 0)

    gnet.share_memory()  # share the global parameters in multiprocessing
    l = "N"  # input("load Y or N:")
    stric = False
    if IS_TEST:
        l = "Y"
        stric = True
    act_net = {}

    if l == "Y":
        act_net = torch.load(f)
        gnet.load_state_dict(act_net, strict=stric)

    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    cp_count = 1
    if IS_TEST:
        print("testing")
    else:
        print("training")
        cp_count = 1

    # parallel training
    if mp.cpu_count() < 6:
        print("cpu alert")
        exit()
    workers = [
        Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, act_net, l, stric, my_q) for
        i in
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
        torch.save(gnet.state_dict(), f)

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

    control = "Y"  # input("control Y or N:")
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
    pref = "control_basea2c_"

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
        if IS_TEST and not os.path.exists(f1):
            print("file:", f1, "does not exist")
            break
        print(f1)
        temp, _ = main(f1, my_q, fname, my_r)
    # name of csv file

    filename = "control_basea2c.csv"

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
    f = open("a2c_tasks123.txt", "w")
    while True:
        r = my_q.get()
        if r is not None:
            print(r)
            mystr = str(r) + "\n"
            f.write(mystr)

        else:
            break
    f.close()
