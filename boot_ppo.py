"""
PPO bootcamp
"""
import torch
from boot_utils.doom_util import v_wrap, tracker, get_dist, \
    in_center3, tracker2, to_center, to_border, breaker, record_boot, helper
import torch.multiprocessing as mp
from boot_utils.Nav import Net as nav
from boot_utils.ppo_util import Agent

import numpy as np
from viz_tasks15 import SailonViz as SViz
import random
import os
import csv
import sys

os.environ["OMP_NUM_THREADS"] = "4"

UPDATE_GLOBAL_ITER = 20
GAMMA = 0.97
MAX_EP = 10#2000
HIDDEN_SIZE = 32
H_SIZE = 16

IS_CONTROL = False
IS_TEST = False

STATE_SIZE = 25
ACTION_SIZE = 4


class my_task():

    def __init__(self, strategist, nav_room, nav_ammo, global_ep, global_ep_r, res_queue, name, global_kills,
                 global_health, global_ammo, test_results, info_list):

        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.strategist = strategist
        self.g_k = global_kills
        self.g_h = global_health
        self.g_a = global_ammo
        self.nav_room = nav_room
        self.nav_ammo = nav_ammo
        self.info_list = info_list
        self.test_results = test_results

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
        self.game_combat = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed, task=1)
        self.game_ammo = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed, task=2)
        self.game_health = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed, task=3)

    def helm(self, nav_vec, state, combat, i_targ, clip, med, act, targ_coord, tir, p_coord, player, ammo):

        m_act = np.dtype('int64').type(7)
        n_act = np.dtype('int64').type(7)
        r_act = np.dtype('int64').type(7)
        c_act = np.dtype('int64').type(7)
        override = False

        if act == 0 and ammo > 0:
            if combat == 1:

                c_act = np.dtype('int64').type(4)

            elif combat == 2:
                c_act = np.dtype('int64').type(5)

            elif combat == 3:
                c_act = np.dtype('int64').type(6)
            elif combat == 4 and ammo > 0:
                c_act = np.dtype('int64').type(0)
            elif combat == 5 and ammo > 0:
                c_act = np.dtype('int64').type(1)
        if act == 2 and med:
            m_coord = tracker(med)
            if m_coord == p_coord:
                if player['angle'] == 90:
                    nav_vec[4] = float(med['x_position'])
                    nav_vec[5] = float(med['y_position'])
                    nav_vec[6] = get_dist(med, player)
                    m_act = self.nav_ammo.choose_action(v_wrap(nav_vec[None, :]))
                else:

                    if 270 > player['angle'] > 90:
                        m_act = np.dtype('int64').type(5)
                    else:
                        m_act = np.dtype('int64').type(4)

            elif p_coord == 1:
                if not in_center3(player) and helper(player, combat, state):

                    m_act = to_border(player, m_coord)

                elif player['angle'] != 315:
                    if 315 > player['angle'] > 135:
                        n_act = np.dtype('int64').type(4)
                    else:
                        n_act = np.dtype('int64').type(5)
                else:
                    nav_vec[4] = self.room_list[m_coord - 1]['x_position']
                    nav_vec[5] = self.room_list[m_coord - 1]['y_position']
                    nav_vec[6] = get_dist(player, self.room_list[m_coord - 1])
                    m_act = self.nav_room.choose_action(v_wrap(nav_vec[None, :]))

            else:
                m_act = to_center(player, p_coord)

                override = True
        if act == 3 and clip:
            a_coord = tracker(clip)

            if a_coord == p_coord:
                if player['angle'] == 90:
                    nav_vec[4] = float(clip['x_position'])
                    nav_vec[5] = float(clip['y_position'])
                    nav_vec[6] = get_dist(player, clip)
                    r_act = self.nav_ammo.choose_action(v_wrap(nav_vec[None, :]))

                else:
                    if 270 > player['angle'] > 90:
                        r_act = np.dtype('int64').type(5)
                    else:
                        r_act = np.dtype('int64').type(4)
            elif p_coord == 1:
                if not in_center3(player) and helper(player, combat, state):
                    r_act = to_border(player, a_coord)

                elif player['angle'] != 315:
                    if 315 > player['angle'] > 135:
                        r_act = np.dtype('int64').type(4)
                    else:
                        r_act = np.dtype('int64').type(5)
                else:
                    nav_vec[4] = self.room_list[a_coord - 1]['x_position']
                    nav_vec[5] = self.room_list[a_coord - 1]['y_position']
                    nav_vec[6] = get_dist(player, self.room_list[a_coord - 1])
                    r_act = self.nav_room.choose_action(v_wrap(nav_vec[None, :]))

            else:  # other rooms
                r_act = to_center(player, p_coord)

                override = True

        if act == 1:

            if tir:

                if player['angle'] != 90:
                    if 270 > player['angle'] > 90:
                        n_act = np.dtype('int64').type(5)
                    else:
                        n_act = np.dtype('int64').type(4)
                elif p_coord == 1:

                    nav_vec[4] = self.patrol_list[i_targ]['x_position']
                    nav_vec[5] = self.patrol_list[i_targ]['y_position']
                    nav_vec[6] = get_dist(player, self.patrol_list[i_targ])

                    n_act = self.nav_ammo.choose_action(v_wrap(nav_vec[None, :]))

                else:
                    nav_vec[4] = self.room_list[p_coord - 1]['x_position']
                    nav_vec[5] = self.room_list[p_coord - 1]['y_position']
                    nav_vec[6] = get_dist(player, self.room_list[p_coord - 1])
                    n_act = self.nav_ammo.choose_action(v_wrap(nav_vec[None, :]))

            elif p_coord != 1:

                n_act = to_center(player, p_coord)

                override = True
            else:
                if not in_center3(player) and helper(player, combat, state):
                    n_act = to_border(player, targ_coord)

                elif player['angle'] != 315:
                    if 315 > player['angle'] > 135:
                        n_act = np.dtype('int64').type(4)
                    else:
                        n_act = np.dtype('int64').type(5)
                else:
                    nav_vec[4] = self.room_list[targ_coord - 1]['x_position']
                    nav_vec[5] = self.room_list[targ_coord - 1]['y_position']
                    nav_vec[6] = get_dist(player, self.room_list[targ_coord - 1])
                    n_act = self.nav_room.choose_action(v_wrap(nav_vec[None, :]))

        return r_act, m_act, c_act, n_act, override

    def run(self):  # bookmark
        """
        Main Driver Method
        :return:
        """
        total_step = 1
        actions = ['left', 'right', 'backward', 'forward', 'shoot', 'turn_left', 'turn_right', 'nothing']
        actions2 = ['left', 'right', 'backward', 'forward', 'turn_left', 'turn_right']
        v_count = 0

        task_index = STATE_SIZE - 1
        reload_index = STATE_SIZE - 2
        combat_index = STATE_SIZE - 3
        heal_index = STATE_SIZE - 4
        task_var = 1.0
        pref_list = np.zeros([MAX_EP])

        raw_list = np.zeros([MAX_EP])
        turn = 0
        switch = 4
        seed = 0
        episode = 0
        pref_total = 0.0
        if self.name == "w01":
            task_var = 1.0
            turn = 3
        if self.name == "w02":
            task_var = 2.0
            turn = 6
        if self.name == "w03":
            task_var = 3.0
            turn = 9

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
                    turn = 4  # 6
                    task_var = 2.0
                    game = self.game_ammo

                elif task == 3:
                    turn = 8  # 3
                    task_var = 3.0
                    game = self.game_health

            state = game.reset(seed)

            player = state['player']
            test_obst = state['items']['obstacle']

            state_vec, nav_vec, e_count, combat, clip, med, targ_coord, tir, seek = breaker(state,
                                                                                            test_obst)

            over_ride = False
            health = int(player['health'])

            t_count = e_count
            kills = 0
            buffer_s, buffer_a, buffer_r = [], [], []

            pl_x = player['x_position']
            pl_y = player['y_position']
            ammo = int(player['ammo'])
            patrol_targ = 0
            a_count = 0
            s_count = 0
            h_count = 0
            prev_act = 'nothing'
            rotate_count = 0

            ep_reward = 0.0
            stuck = False
            p_coord = tracker(player)

            state_vec[task_index] = task_var

            if e_count > 0:  # can attack
                a_check = seek
                if combat > 0:
                    if a_check:
                        state_vec[combat_index] = 3.0
                    else:
                        state_vec[combat_index] = 2.0
                elif a_check:
                    state_vec[combat_index] = 1.0

            if med and ACTION_SIZE == 4:
                if tracker(med) == p_coord and get_dist(med, player) <= 200:
                    state_vec[heal_index] = 2.0
                else:
                    state_vec[heal_index] = 1.0

            if clip:
                if tracker(clip) == p_coord and get_dist(clip, player) <= 200:
                    state_vec[reload_index] = 2.0
                else:
                    state_vec[reload_index] = 1.0

            while True:
                step += 1
                reward = -1
                p_coord = tracker(player)
                act, prob, val = self.strategist.choose_action(state_vec)

                r_act, m_act, c_act, n_act, override = self.helm(nav_vec, state, combat, patrol_targ, clip, med, act,
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
                    elif act == 1:  # navigation
                        my_act = actions2[n_act]
                    elif act == 2:  # health skill
                        if m_act < 7:
                            if turn > 7:
                                reward += 0.5
                            my_act = actions2[m_act]

                        else:

                            my_act = 'nothing'
                            immobile = True
                            reward = -2
                    else:  # ammo skill
                        if r_act < 7:
                            my_act = actions2[r_act]
                            if switch <= turn <= 7:
                                reward += 0.5

                        else:
                            my_act = 'nothing'
                            immobile = True
                            reward = -2
                    if not immobile and stuck:

                        act_temp = prev_act
                        while act_temp == prev_act:
                            act_temp = actions2[np.random.randint(0, 4)]

                            my_act = act_temp

                        stuck = False

                if my_act == 'turn_left' or my_act == 'turn_right':
                    rotate_count += 1
                else:
                    rotate_count = 0

                if rotate_count >= 10:
                    stuck = True

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                new_state, performance, done, victory, dead = game.step(my_act)

                player = new_state['player']

                pl_x2 = player['x_position']
                pl_y2 = player['y_position']

                n_health = int(player['health'])
                items = new_state['items']
                n_ammo = int(player['ammo'])
                nstate_vec, nnav_vec, e_temp, combat, clip, med, targ_coord, tir, seek = breaker(new_state, test_obst)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if get_dist(player, self.patrol_list[patrol_targ]) <= 60.0:
                    patrol_targ += 1
                    if patrol_targ == 4:
                        patrol_targ = 0

                if len(items['ammo']) <= 0 and switch <= turn <= 7:
                    done = True
                    victory = True
                if len(items['health']) <= 0 and turn > 7:
                    done = True
                    victory = True

                if int(pl_x) == int(pl_x2) and int(pl_y) == int(pl_y2):
                    if my_act == 'left' or my_act == 'right' or my_act == 'backward' or my_act == 'forward':
                        stuck = True
                        prev_act = my_act

                if victory:
                    v_count += 1
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
                    reward += 10
                    if act == 3:
                        reward += 50
                        s_count += 1
                    if task_var == 2.0:
                        reward += 15
                    a_count += 1

                if n_health > health:

                    reward += 10
                    if act == 2:
                        reward += 50
                    if task_var == 3.0:
                        reward += 15
                    h_count += 1

                elif n_health < health:
                    reward -= 1

                ammo = n_ammo
                health = n_health

                nstate_vec[task_index] = task_var

                if e_count > 0:
                    a_check = seek
                    if combat > 0:
                        if a_check:
                            nstate_vec[combat_index] = 3.0
                        else:
                            nstate_vec[combat_index] = 2.0
                    elif a_check:
                        nstate_vec[combat_index] = 1.0

                if med and ACTION_SIZE == 4:
                    if tracker(med) == p_coord and get_dist(med, player) <= 200:
                        nstate_vec[heal_index] = 2.0
                    else:
                        nstate_vec[heal_index] = 1.0

                if clip:
                    if tracker(clip) == p_coord and get_dist(clip, player) <= 200:
                        nstate_vec[reload_index] = 2.0
                    else:
                        nstate_vec[reload_index] = 1.0
                ep_reward += reward
                if not IS_TEST:
                    buffer_a.append(act)
                    buffer_s.append(state_vec)
                    buffer_r.append(reward)
                    self.strategist.remember(state_vec, act, prob, val, reward, done)
                if (
                        not IS_TEST and total_step % UPDATE_GLOBAL_ITER == 0) or done:  # update global and assign to local net
                    if not IS_TEST:
                        self.strategist.learn()

                    buffer_s, buffer_a, buffer_r = [], [], []

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
                            record_boot(self.g_ep, self.g_ep_r, performance, self.res_queue, self.name, t_count,
                                        kills,
                                        victory,
                                        dead, a_count, h_count, task_var, self.g_k, self.g_h, self.g_a,
                                        MAX_EP)
                        break
                state_vec = nstate_vec
                state = new_state
                nav_vec = nnav_vec
                total_step += 1
            turn += 1
        if IS_TEST:
            self.test_results.put([v_count, np.average(raw_list), np.average(pref_list)])
        self.res_queue.put(None)
        self.info_list.put(None)


def train_agent(base_file, test_results, my_res, raw_file, tdir):
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    myshape = np.zeros(STATE_SIZE)
    my_info = mp.Queue()

    strategist = Agent(n_actions=ACTION_SIZE, input_dims=myshape.shape, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
    nav_room = nav(13, 6)
    nav_object = nav(13, 6)

    nav_room.load_state_dict(torch.load("weights/nav_room.txt"))
    nav_object.load_state_dict(torch.load("weights/nav_item.txt"))
    l = "N"
    if IS_TEST:
        l = "Y"


    if l == "Y":
        strategist.load_weights(base_file, tdir)


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
    worker = my_task(strategist, nav_room, nav_object, global_ep, global_ep_r, res_queue, 0, global_kills, global_health, global_ammo,
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
    with open(raw_file, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        csvwriter.writerows(myinfo)
    csvfile.close()

    m_kills = 250
    m_items = 150

    if not IS_TEST and (IS_CONTROL or (
            global_kills.value > m_kills and global_health.value > m_items and global_ammo.value > m_items)):
        strategist.save_weights(base_file, tdir)

        my_r2 = np.add(my_res, res)

        return True, my_r2

    if IS_TEST:
        return True, my_res
    return False, my_res


if __name__ == "__main__":
    starting_index = 0
    agent_count = 2#30
    test_ep = 10#1000

    n = len(sys.argv)
    control = "Y"
    #t_dir = "4"
    if n == 2:
        control = sys.argv[1]

    else:
        print("invalid arguments need control, task")


    testing = "N"
    is_load = "N"

    if control == "Y" or control == "y":
        IS_CONTROL = True
    if testing == "Y" or testing == "y":
        IS_TEST = True

    if IS_TEST:
        MAX_EP = test_ep

    test_results = mp.Queue()
    my_res = np.zeros([MAX_EP])
    fname = "ppoboot_"
    if IS_CONTROL:
        fname = "ppocontrol_"
    tdir = "tasks123"

    for ind in range(agent_count):
        n = ind + starting_index
        f_temp = fname +"task123_"+ str(n)
        base_file = f_temp + ".txt"
        raw_file = tdir + "/" + f_temp + "raw.csv"

        while True:
            temp, my_res = train_agent(base_file, test_results, my_res, raw_file, tdir)
            if temp:

                break
            else:
                print("retraining")

    IS_TEST = True

    if IS_TEST:
        MAX_EP = test_ep
    test_results = mp.Queue()

    for ind in range(agent_count):
        n = ind + starting_index
        f_temp = fname + "task123" + "_" + str(n)
        base_file = f_temp + ".txt"
        raw_file = tdir + "/" + f_temp + "rawtest.csv"
        temp, _ = train_agent(base_file, test_results, my_res, raw_file, tdir)
    # name of csv file
    filename = "results/boot_ppo.csv"

    if IS_CONTROL:
        filename = "results/control_ppo.csv"

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

    f = open("results/myout_task123_ppo.txt", "w")
    if IS_CONTROL:
        f.write("control\n")
    else:
        f.write("boot\n")
    while True:
        r = test_results.get()
        if r is not None:
            print(r)
            mystr = str(r) + "\n"
            f.write(mystr)

        else:
            break

    f.close()
    print("done")
