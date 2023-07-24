
from boot_utils.doom_util import record_boot, tracker, get_dist, break_ammo, \
    break_health, get_angle, h_check

import torch.multiprocessing as mp
from boot_utils.ppo_util import Agent

import numpy as np
from viz_tasks15 import SailonViz as SViz
import random
import os
import csv

os.environ["OMP_NUM_THREADS"] = "4"

UPDATE_GLOBAL_ITER = 20
GAMMA = 0.97  # 0.60  # 0.97
MAX_EP = 4
HIDDEN_SIZE = 32  # 128
H_SIZE = 16  # 64

IS_CONTROL = False
IS_TEST = False

STATE_SIZE = 28
ACTION_SIZE = 7



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

    def __init__(self, strategist, global_ep, global_ep_r, res_queue, name, global_kills,
                 global_health, global_ammo, test_results, info_list):

        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.strategist = strategist
        self.g_k = global_kills
        self.g_h = global_health
        self.g_a = global_ammo
        self.info_list = info_list
        self.test_results = test_results




        seed = 97

        self.seed_list = []
        self.use_seed = False

        if IS_TEST:
            self.use_seed = True
        # print(seed)
        if self.use_seed:
            random.seed(seed)
            np.random.seed(seed)


            self.seed_list = [np.random.randint(0, 1000) for i in range(MAX_EP)]

        use_mock = 0
        use_novel = 0
        level = 0
        difficulty = 'easy'

        self.game_combat = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed, task=1)
        self.game_ammo = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed, task=2)
        self.game_health = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed, task=3)

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

        task_index = STATE_SIZE - 1
        combat_index = STATE_SIZE - 2


        task_var = 1.0
        pref_list = np.zeros([MAX_EP])

        raw_list = np.zeros([MAX_EP])
        turn = 0
        switch = 4
        seed = 0
        episode = 0

        pref_total = 0.0
        game = None

        while (not IS_TEST and self.g_ep.value < MAX_EP) or (IS_TEST and episode < MAX_EP):
            step = 0

            if not IS_TEST and not IS_CONTROL:

                if turn < switch:
                    task_var = 1.0
                    game = self.game_combat

                else:
                    game = self.game_ammo
                    task_var = 2.0

                    if turn > 7:
                        task_var = 3.0
                        game = self.game_health
                        if turn > 11:
                            turn = 0
                            task_var = 1.0

                            game = self.game_combat
            if IS_TEST:
                seed = self.seed_list[episode]

            if IS_TEST or IS_CONTROL:

                if IS_TEST:
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
            state_vec, e_count, e_list = breaker(state)

            health = int(player['health'])

            t_count = e_count
            kills = 0

            pl_x = player['x_position']
            pl_y = player['y_position']
            ammo = int(player['ammo'])
            a_count = 0
            s_count = 0
            h_count = 0


            state_vec[task_index] = task_var
            state_vec[combat_index] = self.check_shoot(state, walls)

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



                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                new_state, performance, done, victory, dead = game.step(my_act)

                player = new_state['player']

                pl_x2 = player['x_position']
                pl_y2 = player['y_position']

                n_health = int(player['health'])
                items = new_state['items']
                n_ammo = int(player['ammo'])
                nstate_vec, e_temp, elist = breaker(new_state)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


                if len(items['ammo']) <= 0 and switch <= turn <= 7:
                    done = True
                    victory = True
                if len(items['health']) <= 0 and turn > 7:  # 12:
                    done = True
                    victory = True

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

                    self.strategist.remember(state_vec, act, prob, val, reward, done)
                if (
                        not IS_TEST and total_step % UPDATE_GLOBAL_ITER == 0) or done:  # update global and assign to local net
                    # sync
                    if not IS_TEST:
                        self.strategist.learn()

                    if done:  # done and print information

                        if IS_TEST:
                            pref_list[episode] = performance
                            raw_list[episode] = ep_reward
                            self.info_list.put([episode, ep_reward, step, performance, kills, a_count, h_count])

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

                            record_boot(self.g_ep, self.g_ep_r, performance, self.res_queue, self.name, t_count, kills,
                                        victory,
                                        dead, a_count, h_count, task_var, self.g_k, self.g_h, self.g_a, MAX_EP)
                        break
                state_vec = nstate_vec
                state = new_state
                total_step += 1
            turn += 1
        if IS_TEST:


            self.test_results.put([v_count, np.average(raw_list), np.average(pref_list)])
        self.res_queue.put(None)
        self.info_list.put(None)


def train_agent(base_file, test_results, my_res, new_file, raw_file, tdir):
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    myshape = np.zeros(STATE_SIZE)
    my_info = mp.Queue()

    strategist = Agent(n_actions=ACTION_SIZE, input_dims=myshape.shape, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)

    l = "N"
    if IS_TEST:
        l = "Y"


    if l == "Y":
        strategist.load_weights(base_file, tdir)
    print(raw_file)

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    global_kills = mp.Value('i', 0)
    global_health = mp.Value('i', 0)
    global_ammo = mp.Value('i', 0)
    if IS_TEST:
        print("testing")
    else:
        print("training")


    worker = Worker(strategist, global_ep, global_ep_r, res_queue, 0, global_kills, global_health, global_ammo,
                     test_results, my_info)
    worker.run()
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
    print(raw_file)
    with open(raw_file, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        csvwriter.writerows(myinfo)
    csvfile.close()

    print(base_file)
    print(tdir)
    if not IS_TEST:
        strategist.save_weights(base_file, tdir)

        my_r2 = np.add(my_res, res)

        return True, my_r2

    if IS_TEST:
        return True, my_res
    return False, my_res

if __name__ == "__main__":

    # total arguments

    control = "Y"
    isa2c = "N"


    start_index = 0
    agent_count = 2
    test_ep = 10



    is_load = "N"


    cp_count = 5


    test_results = mp.Queue()
    my_res = np.zeros([MAX_EP])
    train_metrics = []

    fname = "base_ppo_"

    #fname = "tasks123/" + fname
    for ind in range(agent_count):
        n = ind + start_index
        f_temp = fname + str(n)
        base_file = f_temp + ".txt"
        new_file = fname + "tasks123_" + str(n) + ".txt"
        raw_file = "tasks123/"+f_temp + "raw.csv"

        print(base_file)


        temp, my_res = train_agent(base_file, test_results, my_res, new_file, raw_file, "tasks123")


    IS_TEST = True
    cp_count = 1
    MAX_EP = test_ep

    test_results = mp.Queue()
    new_file = "dud.txt"

    for ind in range(agent_count):
        n = ind + start_index
        f_temp = fname + str(n)
        base_file = fname + str(n) + ".txt"
        raw_file = "tasks123/" + f_temp + "tasks123_rawtest.csv"

        print(base_file)

        _, _ = train_agent(base_file, test_results, my_res, new_file, raw_file, "tasks123")
    # name of csv file
    filename = "base_ppo_task123.csv"
    outname = "base_ppo_task123.txt"
    first_line = "base\n"


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
