"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
import torch
from boot_utils.doom_util import v_wrap, push_and_pull, record_base, tracker, get_dist, break_ammo, \
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
MAX_EP = 2000
HIDDEN_SIZE = 64
H_SIZE = 32

IS_CONTROL = False
IS_TEST = False

STATE_SIZE = 28
ACTION_SIZE = 7


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

    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, f, l, stric, test_results, info_list):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gstrat, self.opt = gnet, opt
        self.lstrat = nav(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, H_SIZE)
        self.test_results = test_results
        self.info_list = info_list
        if l == "Y":
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

        self.game_combat = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed, task=1, base=True)
        self.game_ammo = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed, task=2, base=True)
        self.game_health = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed, task=3, base=True)

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

        task_index = STATE_SIZE - 1
        combat_index = STATE_SIZE - 2
        task_var = 1.0
        pref_list = np.zeros([MAX_EP])

        raw_list = np.zeros([MAX_EP])
        turn = 0
        switch = 4
        seed = 0
        episode = 0

        if self.name == "w01":
            task_var = 1.0
            turn = 3
        if self.name == "w02":
            task_var = 2.0
            turn = 6
        if self.name == "w03":
            task_var = 3.0
            turn = 9
        pref_total = 0.0
        game = None

        while (not IS_TEST and self.g_ep.value < MAX_EP) or (IS_TEST and episode < MAX_EP):
            step = 0

            if IS_TEST:
                seed = self.seed_list[episode]
                np.random.seed(seed)

            task = np.random.randint(1, 4)
            if task == 1:
                turn = 0
                task_var = 1.0

                game = self.game_combat
            elif task == 2:
                turn = 4
                task_var = 2.0
                game = self.game_ammo

            elif task == 3:
                turn = 8
                task_var = 3.0
                game = self.game_health

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

            state_vec[task_index] = task_var
            state_vec[combat_index] = self.check_shoot(state, walls)
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

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                new_state, performance, done, victory, dead = game.step(my_act)

                player = new_state['player']

                pl_x2 = player['x_position']
                pl_y2 = player['y_position']

                n_health = int(player['health'])
                items = new_state['items']
                n_ammo = int(player['ammo'])
                nstate_vec, e_temp, elist = breaker(new_state)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                if len(items['ammo']) <= 0 and switch <= turn <= 7:
                    done = True
                    victory = True
                if len(items['health']) <= 0 and turn > 7:
                    done = True
                    victory = True

                if int(pl_x) == int(pl_x2) and int(pl_y) == int(pl_y2):
                    if my_act == 'left' or my_act == 'right' or my_act == 'backward' or my_act == 'forward' or my_act == 'nothing':
                        reward -= 1

                for key in elist.keys():

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
                    e_count -= 1
                    kills += 1
                    if step < 100:
                        reward += 20
                    elif step < 400:
                        reward += 10

                if n_ammo > ammo:
                    reward += 75
                    if task_var == 2.0:
                        reward += 15
                    a_count += 1

                if n_health > health:

                    reward += 75
                    if task_var == 3.0:
                        reward += 15
                    h_count += 1

                elif n_health < health:
                    reward -= 1

                ammo = n_ammo
                health = n_health

                nstate_vec[task_index] = task_var
                nstate_vec[combat_index] = self.check_shoot(new_state, walls)
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
                            self.info_list.put([self.g_ep.value, ep_reward, step, performance, kills, a_count, h_count])

                            task = "combat"

                            if task_var == 2:
                                task = "reload"
                            elif task_var == 3:
                                task = "heal"
                            episode += 1
                            pref_total += performance

                            print(
                                self.name,
                                "Ep:", episode, "enemies:", t_count, "kills:", kills, "victory:", victory,
                                "dead:", dead, "ammo:", a_count,
                                "health:", h_count,
                                "| Ep_r: %.2f" % (pref_total / episode), " indiv: %.2f" % performance, task
                            )


                        else:
                            self.info_list.put([self.g_ep.value, ep_reward, step, performance, kills, a_count, h_count])

                            record_base(self.g_ep, self.g_ep_r, performance, self.res_queue, self.name, t_count, kills,
                                        victory,
                                        dead, a_count, h_count, task_var)
                        break
                state_vec = nstate_vec
                state = new_state
                total_step += 1
            turn += 1
        if IS_TEST:
            self.test_results.put([v_count, np.average(raw_list), np.average(pref_list)])
        self.res_queue.put(None)
        self.info_list.put(None)


def train_agent(base_file, test_results, my_res, new_file, raw_file, cp_count):
    gstrat = nav(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, H_SIZE)  # global network

    my_info = mp.Queue()
    l = "N"
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

    workers = [
        Worker(gstrat, opt, global_ep, global_ep_r, res_queue, i, act_net, l, stric, test_results, my_info)
        for
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
        return True, my_res2

    return False, my_res


if __name__ == "__main__":

    # total arguments
    n = len(sys.argv)
    control = "Y"
    isa2c = "N"
    if n == 2:
        isa2c = sys.argv[1]
    else:
        print("invalid arguments need control, is_a2c")

    start_index = 0
    agent_count = 2
    test_ep = 10
    nav_room = nav(13, 6)
    nav_item = nav(13, 6)



    is_load = "N"
    is_a2c = False
    if isa2c == "Y":
        is_a2c = True

    cp_count = 5
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
    fname = "tasks123/" + fname
    for ind in range(agent_count):
        n = ind + start_index
        f_temp = fname + str(n)
        base_file = f_temp + ".txt"
        new_file = fname + "task123_" + str(n) + ".txt"
        raw_file = f_temp + "raw.csv"

        print(base_file)
        while True:

            temp, my_res = train_agent(base_file, test_results, my_res, new_file, raw_file, cp_count)
            if temp:
                break

    IS_TEST = True
    cp_count = 1
    MAX_EP = test_ep

    test_results = mp.Queue()
    new_file = "dud.txt"

    for ind in range(agent_count):
        n = ind + start_index
        f_temp = fname + str(n)
        base_file = fname + "task123_" + str(n) + ".txt"
        raw_file = f_temp + "task123_raw.csv"
        if not os.path.exists(base_file):
            print("file:", base_file, "does not exist")
            break
        print(base_file)

        _, _ = train_agent(base_file, test_results, my_res, new_file, raw_file, cp_count)
    # name of csv file
    filename = "base_task123.csv"
    outname = "base_task123.txt"
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
                    agent_count += h[0]
                if lines and not header:
                    l = np.asarray(lines, dtype="float64")
                    my_res = np.add(my_res, l)

                header = False

    # writing to csv file

    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        head = np.zeros([len(my_res)])

        head[0] = agent_count
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

    f.close()
    print("done")
