"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
from boot_utils.doom_util import record_fell, tracker, get_dist, break_ammo, \
    break_health, get_angle, h_check
import sys
import torch.multiprocessing as mp
from boot_utils.ppo_util import Agent

import numpy as np
from viz_tasks15 import SailonViz as SViz

import random
import os
import csv

os.environ["OMP_NUM_THREADS"] = "4"

UPDATE_GLOBAL_ITER = 20
MAX_EP = 10

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


def breaker(state):  # bookmark
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
                     h_count] + strat_health + strat_obst + [0.0, 0, 0]  # wall +[0.0]


    return np.asarray(sensor_vec), e_count, e_list


class Worker():
    def __init__(self, strategist, global_ep, global_ep_r, res_queue, name, test_results, my_jump, my_asym, info_list):

        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.strategist = strategist

        self.test_results = test_results
        self.my_jump = my_jump
        self.my_asym = my_asym
        self.info_list = info_list
        self.room_list = [{"x_position": 0.0, "y_position": 0.0}, {"x_position": 0.0, "y_position": 458.0},
                          {"x_position": 0.0, "y_position": -458.0}, {"x_position": 458.0, "y_position": 0.0},
                          {"x_position": -458.0, "y_position": 0.0},
                          {"x_position": 0.0, "y_position": 0.0}]
        self.patrol_list = [{"x_position": 180.0, "y_position": 0.0}, {"x_position": 0.0, "y_position": 180.0},
                            {"x_position": -180.0, "y_position": 0}, {"x_position": 0, "y_position": -180.0}]

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
        r_list = np.zeros([MAX_EP])

        r_list2 = np.zeros([MAX_EP])
        turn = 0
        switch = 4
        s = 0
        episode = 0

        my_av = 0.0
        game = None

        while (not IS_TEST and self.g_ep.value < MAX_EP) or (IS_TEST and episode < MAX_EP):
            step = 0
            turn = 0
            if IS_TEST:
                s = self.seed_list[episode]

            game = self.game

            state = game.reset(s)

            player = state['player']

            walls = state['walls']
            state_vec, e_count, e_list = breaker(state)
            # initial state_vec

            hp = int(player['health'])

            t_count = e_count
            kills = 0

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
                act, prob, val = self.strategist.choose_action(state_vec)

                my_act = actions[act]
                if my_act == "shoot":
                    if ammo > 0:
                        fired = True

                new_state, performance, done, victory, dead = game.step(my_act)

                player = new_state['player']

                pl_x2 = player['x_position']
                pl_y2 = player['y_position']

                health = int(player['health'])
                items = new_state['items']
                n_ammo = int(player['ammo'])
                nstate_vec, e_temp, elist = breaker(new_state)

                if int(pl_x) == int(pl_x2) and int(pl_y) == int(pl_y2):
                    if my_act == 'left' or my_act == 'right' or my_act == 'backward' or my_act == 'forward':
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

                    a_count += 1

                if health > hp:

                    reward += 75

                    h_count += 1

                elif health < hp:
                    reward -= 1

                ammo = n_ammo
                hp = health

                nstate_vec[test_index] = task_var
                nstate_vec[sight_index] = self.check_shoot(new_state, walls)

                ep_reward += reward
                if not IS_TEST:
                    self.strategist.remember(state_vec, act, prob, val, reward, done)
                if (
                        not IS_TEST and total_step % UPDATE_GLOBAL_ITER == 0) or done:  # update global and assign to local net
                    # sync
                    if not IS_TEST:
                        self.strategist.learn()

                    if done:  # done and print information

                        if IS_TEST:
                            r_list[episode] = performance
                            r_list2[episode] = ep_reward
                            task = "combat"
                            if task_var == 2:
                                task = "reload"
                            elif task_var == 3:
                                task = "heal"
                            episode += 1
                            self.info_list.put([self.g_ep.value, ep_reward, step, performance, kills, a_count, h_count])
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

        if IS_TEST:
            self.test_results.put([v_count, np.average(r_list2), np.average(r_list)])
        self.res_queue.put(None)
        self.my_jump.put(None)
        self.my_asym.put(None)
        self.info_list.put(None)

def train_agent(base_file, test_results, my_res, new_file, train_metrics, raw_file, bdir, tdir):
    """
    runs a single game
    :param base_file: file we load from
    :param test_results: test wins, raw score, preformance
    :param my_res: preformance for training
    :param new_file: file we save to
    :param train_metrics: jump start and asympotic performance queue
    :param raw_file: file to save episode info to
    :return:
    """
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    myshape = np.zeros(STATE_SIZE)
    strategist = Agent(n_actions=ACTION_SIZE, input_dims=myshape.shape, batch_size=batch_size, alpha=alpha,
                       n_epochs=n_epochs)

    my_jump = mp.Queue()
    my_asym = mp.Queue()
    my_info = mp.Queue()

    strategist.load_weights(base_file, bdir)

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    if IS_TEST:
        print("testing")
    else:
        print("training")

    worker = Worker(strategist,  global_ep, global_ep_r, res_queue, 0, test_results, my_jump, my_asym,
                    my_info)
    worker.run()
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

    if not IS_TEST:
        strategist.save_weights(new_file, tdir)

        my_res2 = np.add(my_res, res)

        temp = train_metrics
        temp.append([m_jump, m_asym])
        # TODO possibly return temp for consistency with task45_a3c.py although both ways work
        return my_res2

    return my_res


if __name__ == "__main__":

    n = len(sys.argv)
    control = "Y"
    t_dir = "5"
    if n == 3:
        t_dir = sys.argv[2]
    else:
        print("invalid arguments need control, task")

    starting_index = 0
    num_agents = 2
    test_ep = 10

    testing = "N"  # input("test Y or N:")
    is_load = "N"  # input("continue Y or N:")
    is_control = False
    if control == "Y" or control == "y":
        is_control = True
    if testing == "Y" or testing == "y":
        IS_TEST = True

    if IS_TEST:
        MAX_EP = test_ep

    test_results = mp.Queue()
    my_res = np.zeros([MAX_EP])
    train_metrics = []
    fname = "base_ppo_"
    tdir = "task4"
    if t_dir == "5":
        TASK_5 = True
        tdir = "task5"
    fname2 = tdir + "/" + fname

    for ind in range(num_agents):
        n = ind + starting_index

        f_temp = fname + str(n)
        f_temp2 = fname2 + tdir + str(n)
        base_file = f_temp + ".txt"
        new_file = fname + tdir + "_" + str(n) + ".txt"
        raw_file = f_temp2 + "raw.csv"

        my_res = train_agent(base_file, test_results, my_res, new_file, train_metrics, raw_file, "tasks123", tdir)

    IS_TEST = True

    if IS_TEST:
        MAX_EP = test_ep
    test_results = mp.Queue()
    new_file = "dud.txt"
    for ind in range(num_agents):
        n = ind + starting_index

        f_temp = fname2 + str(n)
        base_file = fname + tdir + "_" + str(n) + ".txt"
        raw_file = f_temp + tdir + "_rawtest.csv"

        print(base_file)
        _ = train_agent(base_file, test_results, my_res, new_file, train_metrics, raw_file, tdir, tdir)
    # name of csv file
    filename = "base_ppo_" + tdir + ".csv"
    outname = "base_ppo_" + tdir + ".txt"
    first_line = "ppo_base\n"

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
                    num_agents += h[0]
                if lines and not header:
                    l = np.asarray(lines, dtype="float64")
                    my_res = np.add(my_res, l)

                header = False

    # writing to csv file

    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        head = np.zeros([len(my_res)])

        head[0] = num_agents
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

