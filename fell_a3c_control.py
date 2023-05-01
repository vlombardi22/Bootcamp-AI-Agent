"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
import torch

from boot_utils.doom_util import v_wrap, push_and_pull, record_fell, tracker, get_dist, break_ammo, \
    break_health, get_angle, h_check
import torch.multiprocessing as mp
from boot_utils.shared import SharedAdam

from boot_utils.Nav import Net as nav
import numpy as np
from viz_tasks15 import SailonViz as SViz

import random
import os
import csv
import sys

os.environ["OMP_NUM_THREADS"] = "4"

UPDATE_GLOBAL_ITER = 20
GAMMA = 0.97
MAX_EP = 1000
HIDDEN_SIZE = 64
H_SIZE = 32

IS_CONTROL = False
IS_TEST = False

STATE_SIZE = 28
ACTION_SIZE = 7

TASK_5 = False


def break_obstacles(items, player):
    min_dist = 10000
    m_obst = None

    for o in items['obstacle']:
        dist = get_dist(player, o)

        if min_dist > dist:
            min_dist = dist
            m_obst = o

    if not m_obst:
        strat_obst = [0.0, 0.0, -1.0, 0.0]

    else:
        angle, _ = get_angle(m_obst, player, 0.0)
        angle = angle * 180 / np.pi
        strat_obst = [float(m_obst['x_position']), float(m_obst['y_position']), min_dist, angle]

    return strat_obst


def break_enemy(enemies, player):
    min_dist = 10000
    m_enemy = None
    elist = {}
    for e in enemies:
        dist = get_dist(player, e)
        elist[int(e['id'])] = h_check(e)
        if min_dist > dist:
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


def breaker(state):
    enemies = state['enemies']
    items = state['items']
    player = state['player']

    strat_enemy, e_list = break_enemy(enemies, player)
    p_coord = tracker(player)
    strat_obst = break_obstacles(items, player)

    e_count = len(enemies)
    a_count = len(items['ammo'])
    h_count = len(items['health'])

    _, strat_ammo = break_ammo(items, player, p_coord, enemies)
    _, strat_health = break_health(items, player, p_coord, enemies)
    sensor_vec = [float(player['x_position']), float(player['y_position']), float(player['angle']), int(player['ammo']),
                  int(player['health']), e_count] + strat_enemy + [a_count] + strat_ammo + [
                     h_count] + strat_health + strat_obst + [0.0, 0, 0]

    return np.asarray(sensor_vec), e_count, e_list


class Worker(mp.Process):

    def __init__(self, gstrat, opt, global_ep, global_ep_r, res_queue, name, f, stric, test_results, my_jump, my_asym,
                 info_list):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gstrat, self.opt = gstrat, opt
        self.lstrat = nav(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, H_SIZE)
        self.test_results = test_results
        self.info_list = info_list
        self.my_jump = my_jump
        self.my_asym = my_asym

        self.lstrat.load_state_dict(f, strict=stric)

        seed = 97

        self.seed_list = []
        self.use_seed = False

        if IS_TEST:
            self.use_seed = True
        if self.use_seed:
            random.seed(seed)
            np.random.seed(seed)

            self.seed_list = [np.random.randint(0, 1000) for i in range(MAX_EP)]

        use_mock = 0
        use_novel = 0
        level = 0
        difficulty = 'easy'
        if TASK_5:
            use_novel = 1

            level = 208

        self.game = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed, task=4, base=True)

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

        total_step = 1
        actions = ['left', 'right', 'backward', 'forward', 'shoot', 'turn_left', 'turn_right', 'nothing']
        v_count = 0

        test_index = STATE_SIZE - 1
        sight_index = STATE_SIZE - 2
        task_var = 0.0
        pref_list = np.zeros([MAX_EP])

        raw_list = np.zeros([MAX_EP])
        turn = 0
        seed = 0
        episode = 0

        my_av = 0.0
        game = self.game

        while (not IS_TEST and self.g_ep.value < MAX_EP) or (IS_TEST and episode < MAX_EP):
            step = 0

            if IS_TEST:
                seed = self.seed_list[episode]
                np.random.seed(seed)

            state = game.reset(seed)

            player = state['player']

            walls = state['walls']

            state_vec, e_count, e_list = breaker(state)  # initial state_vec

            health = int(player['health'])

            t_count = e_count
            kills = 0
            buffer_s, buffer_a, buffer_r = [], [], []

            pl_x = player['x_position']
            pl_y = player['y_position']
            ammo = int(player['ammo'])
            a_count = 0
            h_count = 0

            state_vec[test_index] = task_var
            state_vec[sight_index] = self.check_shoot(state, walls)
            ep_reward = 0.0
            while True:
                step += 1
                reward = -1
                fired = False
                act = self.lstrat.choose_action(v_wrap(state_vec[None, :]))
                my_act = actions[act]
                if my_act == "shoot":
                    if ammo > 0:
                        fired = True

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                new_state, performance, done, victory, dead = game.step(my_act)

                player = new_state['player']

                pl_x2 = player['x_position']
                pl_y2 = player['y_position']

                n_health = int(player['health'])
                n_ammo = int(player['ammo'])
                nstate_vec, e_temp, elist = breaker(new_state)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                if int(pl_x) == int(pl_x2) and int(pl_y) == int(pl_y2):
                    if my_act == 'left' or my_act == 'right' or my_act == 'backward' or my_act == 'forward' or my_act == 'nothing':
                        reward -= 1

                for key in elist.keys():
                    if key in e_list.keys():
                        if elist[key] < e_list[key]:

                            if fired:
                                reward += 40
                e_list = elist

                if victory:
                    v_count += 1
                    reward += 200
                    if step < 751:
                        reward += 10

                pl_x = pl_x2
                pl_y = pl_y2

                if e_temp < e_count:
                    reward += 100
                    kills += 1
                    if step < 100:
                        reward += 20
                    elif step < 400:
                        reward += 10
                e_count = e_temp
                if n_ammo > ammo:
                    reward += 75

                    a_count += 1

                if n_health > health:

                    reward += 75

                    h_count += 1

                elif n_health < health:
                    reward -= 1

                ammo = n_ammo
                health = n_health

                nstate_vec[test_index] = task_var
                nstate_vec[sight_index] = self.check_shoot(new_state, walls)
                ep_reward += reward
                if not IS_TEST:
                    buffer_a.append(act)
                    buffer_s.append(state_vec)
                    buffer_r.append(reward)

                if (
                        not IS_TEST and total_step % UPDATE_GLOBAL_ITER == 0) or done:  # update global and assign to local net
                    # sync

                    if len(buffer_s) > 0 and not IS_TEST:
                        push_and_pull(self.opt, self.lstrat, self.gstrat, done, nstate_vec, buffer_s, buffer_a,
                                      buffer_r,
                                      GAMMA)

                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information

                        if IS_TEST:
                            pref_list[episode] = performance
                            raw_list[episode] = ep_reward
                            task = "task4"
                            self.info_list.put([episode, ep_reward, step, performance, kills, a_count, h_count])
                            episode += 1

                            my_av += performance

                            print(
                                self.name,
                                "Ep:", episode, "enemies:", t_count, "kills:", kills, "victory:", victory,
                                "dead:", dead, "ammo:", a_count,
                                "health:", h_count,
                                "| Ep_r: %.2f" % (my_av / episode), " indiv: %.2f" % performance, task
                            )


                        else:
                            self.info_list.put([self.g_ep.value, ep_reward, step, performance, kills, a_count, h_count])

                            record_fell(self.g_ep, self.g_ep_r, performance, self.res_queue, self.name, t_count, kills,
                                        victory,
                                        dead, a_count, h_count, task_var, self.my_jump, self.my_asym)
                        break
                state_vec = nstate_vec
                state = new_state
                total_step += 1
            turn += 1
        if IS_TEST:
            self.test_results.put([v_count, np.average(raw_list), np.average(pref_list)])
        self.res_queue.put(None)
        self.my_jump.put(None)
        self.my_asym.put(None)
        self.info_list.put(None)


def train_agent(base_file, test_results, my_res, new_file, train_metrics, raw_file, cp_count):
    gstrat = nav(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, H_SIZE)  # global network

    my_jump = mp.Queue()
    my_asym = mp.Queue()
    my_info = mp.Queue()
    l = "Y"
    stric = True
    if IS_TEST:
        l = "Y"
        stric = True
    act_net = {}

    if l == "Y":

        act_net = torch.load(base_file)
        gstrat.load_state_dict(act_net, strict=stric)


    opt = SharedAdam(gstrat.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    if IS_TEST:
        print("testing")
    else:
        print("training")


    # parallel training
    if mp.cpu_count() < 6:
        print("cpu alert")
        exit()
    workers = [
        Worker(gstrat, opt, global_ep, global_ep_r, res_queue, i, act_net, stric, test_results, my_jump, my_asym,
               my_info)
        for
        i in
        range(cp_count)]

    [w.start() for w in workers]
    res = []  # record episode reward to plot

    m_jump = 0
    m_asym = 0
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    if not IS_TEST:

        jump = []
        asympt = []

        while True:
            p = my_jump.get()
            if p is not None:
                jump.append(p)
            else:
                break
        m_jump = np.average(jump)

        while True:
            p = my_asym.get()
            if p is not None:
                asympt.append(p)
            else:
                break
        m_asym = np.average(asympt)
    myinfo = []
    while True:
        p = my_info.get()
        if p is not None:
            myinfo.append(p)
        else:
            break
    with open(raw_file, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        csvwriter.writerows(myinfo)
    csvfile.close()
    [w.join() for w in workers]

    if not IS_TEST:
        torch.save(gstrat.state_dict(), new_file)
        my_res2 = np.add(my_res, res)
        my_q3 = train_metrics
        my_q3.append([m_jump, m_asym])
        return my_res2, my_q3
    return my_res, train_metrics


if __name__ == "__main__":
    n = len(sys.argv)
    control = "Y"
    isa2c = "N"
    t_dir = "4"
    if n == 3:
        isa2c = sys.argv[1]
        t_dir = sys.argv[2]
    else:
        print("invalid arguments need control, is_a2c")

    x = 0
    rang = 2
    test_ep = 10



    is_load = "N"
    is_a2c = False
    if isa2c == "Y":
        is_a2c = True
    if control == "Y" or control == "y":
        IS_CONTROL = True
    cp_count = 4
    if is_a2c:
        cp_count = 1

    test_results = mp.Queue()
    my_res = np.zeros([MAX_EP])
    train_metrics = []

    fname = "base_boot_"
    if IS_CONTROL:
        fname = "base_control_"
    if is_a2c:
        fname = fname + "a2c_"
    tdir = "task4"
    if t_dir == "5":
        TASK_5 = True
        tdir = "task5"
    fname2 = tdir + "/" + fname
    fname = "tasks123/" +fname

    for ind in range(rang):
        n = ind + x
        f_temp = fname +"task123_" + str(n)
        f_temp2 = fname2 + tdir + str(n)
        base_file = f_temp + ".txt"
        new_file = fname2 + tdir +"_" + str(n) + ".txt"
        raw_file = f_temp2 + "raw.csv"

        if not os.path.exists(base_file):
            print("file:", base_file, "does not exist")
            break
        print(base_file)

        my_res, train_metrics = train_agent(base_file, test_results, my_res, new_file, train_metrics, raw_file, cp_count)

    IS_TEST = True
    cp_count = 1
    MAX_EP = test_ep

    test_results = mp.Queue()
    new_file = "dud.txt"

    for ind in range(rang):
        n = ind + x
        f_temp = fname2 + str(n)
        base_file = fname2 + tdir +"_" + str(n) + ".txt"
        raw_file = f_temp + "rawtest.csv"


        if not os.path.exists(base_file):
            print("file:", base_file, "does not exist")
            break
        print(base_file)

        _, _ = train_agent(base_file, test_results, my_res, new_file, train_metrics, raw_file, cp_count)  # my_jump, my_asym, my_tran)

    filename = "base_" + tdir + "_a2c.csv"
    outname = "base_" + tdir + "_a2c.txt"
    first_line = "base\n"
    if is_a2c:
        filename = "a2c_" + filename
        outname = "a2c_" + outname
    filename = "results/" + filename
    outname = "results/" + outname
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
                    my_res = np.add(my_res, l)

                header = False

    # writing to csv file

    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        head = np.zeros([len(my_res)])

        head[0] = rang
        rows = [head, my_res]

        csvwriter.writerows(rows)
    csvfile.close()

    test_results.put(None)

    f = open(outname, "w")
    f.write(first_line)
    f.write("wins, raw, pref\n")
    while True:
        r = test_results.get()
        if r is not None:
            mystr = str(r) + "\n"
            f.write(mystr)

        else:
            break
    f.write("jump, asym\n")
    for i in train_metrics:
        mystr2 = str(i) + "\n"
        f.write(mystr2)
    f.close()
    print("done")
"""
if __name__ == "__main__":
    x = 0
    rang = 30
    test_ep = 1000
    control = "Y"
    is_load = "N"

    if control == "Y" or control == "y":
        IS_CONTROL = True

    test_results = mp.Queue()
    train_metrics = []
    my_res = np.zeros([MAX_EP])
    pref = "boot_base2_"
    if IS_CONTROL:
        pref = "control_base2_"

    for ind in range(rang):
        n = ind + x
        f0 = pref + str(n)
        f1 = f0 + ".txt"
        f2 = pref + "task5_" + str(n) + ".txt"
        f3 = f0 + "raw.csv"
        if not os.path.exists(f1):
            print("file:", f1, "does not exist")
            break
        print(f1)

        my_res, train_metrics = main(f1, test_results, my_res, f2, train_metrics, f3)  # my_jump, my_asym, my_tran)

    IS_TEST = True

    MAX_EP = test_ep

    test_results = mp.Queue()
    f2 = "dud.txt"

    for ind in range(rang):
        n = ind + x
        f0 = pref + str(n)
        f1 = pref + "task5_" + str(n) + ".txt"
        f3 = f0 + "rawtest.csv"
        if not os.path.exists(f1):
            print("file:", f1, "does not exist")
            break
        print(f1)

        _, _ = main(f1, test_results, my_res, f2, train_metrics, f3)  # my_jump, my_asym, my_tran)
    # name of csv file
    filename = "control_base_a3c_task5.csv"

    if IS_CONTROL:
        filename = "boot_base_a3c_task5.csv"

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
                    my_res = np.add(my_res, l)

                header = False

    # writing to csv file

    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        head = np.zeros([len(my_res)])

        head[0] = rang
        rows = [head, my_res]

        csvwriter.writerows(rows)
    csvfile.close()

    test_results.put(None)

    f = open("myout_task5_base_a3c.txt", "w")
    # if IS_TEST:
    f.write("boot\n")
    while True:
        r = test_results.get()
        if r is not None:
            print(r)
            mystr = str(r) + "\n"
            f.write(mystr)

        else:
            break
    f.write("other\n")
    print("other")
    for i in train_metrics:
        mystr2 = str(i) + "\n"
        f.write(mystr2)
        print(i)
    f.close()
    print("done")
"""