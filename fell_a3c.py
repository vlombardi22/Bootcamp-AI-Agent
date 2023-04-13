"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
import torch
from doom_util import v_wrap, push_and_pull, record_fell, tracker, get_dist, \
    in_center3, tracker2, to_center, to_border, breaker, helper
import torch.multiprocessing as mp
from Nav import Net as nav
from shared import SharedAdam

import numpy as np
from viz_task45 import SailonViz as SViz

import random
import os
import csv
import sys
os.environ["OMP_NUM_THREADS"] = "4"

UPDATE_GLOBAL_ITER = 20
GAMMA = 0.97  # 0.60  # 0.97
MAX_EP = 20
HIDDEN_SIZE = 32  # 128
H_SIZE = 16  # 64

IS_CONTROL = False
IS_TEST = False

STATE_SIZE = 25
ACTION_SIZE = 4
TASK_5 = False



class Worker(mp.Process):

    def __init__(self, strategist, nav_room, nav_object, opt, global_ep, global_ep_r, res_queue, name, f, stric, test_results, my_jump,
                 my_asym, info_list):
        """
        :param strategist: strategy net
        :param nav_room: navigation skill for rooms
        :param nav_object: navigation skill for objects
        :param opt: optimizer
        :param global_ep: global episode count
        :param global_ep_r: global episode reward
        :param res_queue: training performance
        :param name: thread name
        :param f: file name
        :param stric: load strict
        :param test_results: test metrics
        :param my_jump: jumpstart
        :param my_asym: asymptotic performance
        :param info_list: episode metrics
        """
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gstrat, self.opt = strategist, opt
        self.lstrat = nav(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, H_SIZE)
        self.nav_room = nav_room
        self.nav_object = nav_object
        self.info_list = info_list
        self.test_results = test_results
        self.my_jump = my_jump
        self.my_asym = my_asym
        self.lstrat.load_state_dict(f, strict=stric)

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
        #self.use_seed = 0
        self.use_novel = 0
        level = 0

        if TASK_5:
            self.use_novel = 1

            level = 208
        difficulty = 'easy'

        self.game = SViz(use_mock, self.use_novel, level, False, seed, difficulty, use_seed=self.use_seed)

    def helm(self, nav_vec, state, combat, patrol_targ, clip, med, act, targ_coord, tir, p_coord, player, ammo):
        """
        Function for translating strategist's orders into an actual action
        :param nav_vec: sensor vector for navigation agents
        :param state: current environment state
        :param combat: combat skills action
        :param patrol_targ: search pattern coordinates
        :param clip: most optimal ammo pack
        :param med: most optimal health pack
        :param act: objective picked by skillnet
        :param targ_coord: target coordinates
        :param tir: target in room is the target in the current room
        :param p_coord: the room that the player is currently in
        :param player: player object
        :param ammo: player ammo count
        :return: a usable action
        """
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
                    m_act = self.nav_room.choose_action(v_wrap(nav_vec[None, :]))
                else:

                    if 270 > player['angle'] > 90:
                        m_act = np.dtype('int64').type(5)  # turn right
                    else:
                        m_act = np.dtype('int64').type(4)

            elif p_coord == 1:
                if not in_center3(player) and helper(player, combat, state):

                    m_act = to_border(player, m_coord)

                elif player['angle'] != 315:
                    if 315 > player['angle'] > 135:
                        n_act = np.dtype('int64').type(4)  # turn right
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
                    r_act = self.nav_room.choose_action(v_wrap(nav_vec[None, :]))

                else:
                    if 270 > player['angle'] > 90:
                        r_act = np.dtype('int64').type(5)  # turn right
                    else:
                        r_act = np.dtype('int64').type(4)
            elif p_coord == 1:

                if not in_center3(player) and helper(player, combat, state):
                    r_act = to_border(player, a_coord)

                elif player['angle'] != 315:
                    if 315 > player['angle'] > 135:
                        r_act = np.dtype('int64').type(4)  # turn right
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
                        n_act = np.dtype('int64').type(5)  # turn right
                    else:
                        n_act = np.dtype('int64').type(4)
                elif p_coord == 1:

                    nav_vec[4] = self.patrol_list[patrol_targ]['x_position']
                    nav_vec[5] = self.patrol_list[patrol_targ]['y_position']
                    nav_vec[6] = get_dist(player, self.patrol_list[patrol_targ])

                    n_act = self.nav_room.choose_action(v_wrap(nav_vec[None, :]))

                else:
                    nav_vec[4] = self.room_list[p_coord - 1]['x_position']
                    nav_vec[5] = self.room_list[p_coord - 1]['y_position']
                    nav_vec[6] = get_dist(player, self.room_list[p_coord - 1])
                    n_act = self.nav_room.choose_action(v_wrap(nav_vec[None, :]))

            elif p_coord != 1:

                n_act = to_center(player, p_coord)

                override = True
            else:

                if not in_center3(player) and helper(player, combat, state):
                    n_act = to_border(player, targ_coord)

                elif player['angle'] != 315:
                    if 315 > player['angle'] > 135:
                        n_act = np.dtype('int64').type(4)  # turn left
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
        victory_count = 0

        task_index = STATE_SIZE - 1
        reload_index = STATE_SIZE - 2
        combat_index = STATE_SIZE - 3
        heal_index = STATE_SIZE - 4
        task_var = 0.0
        pref_list = np.zeros([MAX_EP])  # performance

        raw_list = np.zeros([MAX_EP])  # raw reward scores
        seed = 0
        episode = 0

        pref_total = 0.0

        while (not IS_TEST and self.g_ep.value < MAX_EP) or (IS_TEST and episode < MAX_EP):

            if IS_TEST:
                seed = self.seed_list[episode]

            state = self.game.reset(seed)
            pillar = state['items']['obstacle']  # the pillar does not move so we only need to get it once

            state_vec, nav_vec, e_count, combat, clip, med, targ_coord, tir, can_kill = breaker(state,
                                                                                                pillar)  # initial state_vec

            step = 0  # current step
            kills = 0  # kills
            patrol_targ = 0  # coordinate for moving in a rough circle around the central room
            a_count = 0  # number of ammo packs
            h_count = 0  # number of health packs
            prev_act = 'nothing'  # previous action
            stuck = False  # is the agent stuck
            rotate_count = 0  # how many times has the agent turned in a row
            ep_reward = 0.0  # episode reward
            over_ride = False  # override variable for going through doors
            t_count = e_count  # total enemy count at round start

            player = state['player']  # player object
            health = int(player['health'])  # health points or current player health
            pl_x = player['x_position']  # player x
            pl_y = player['y_position']  # player y
            ammo = int(player['ammo'])  # player ammo
            p_coord = tracker(player)  # player room coord
            buffer_s, buffer_a, buffer_r = [], [], []
            state_vec[task_index] = task_var  # what tasks is the agent doing

            if e_count > 0:  # check if the agent can fight
                a_check = can_kill
                if combat > 0:
                    if a_check:
                        state_vec[combat_index] = 3.0
                    else:
                        state_vec[combat_index] = 2.0
                elif a_check:
                    state_vec[combat_index] = 1.0

            if med:  # check if the agent can get health
                if tracker(med) == p_coord and get_dist(med, player) <= 200:
                    state_vec[heal_index] = 2.0
                else:
                    state_vec[heal_index] = 1.0

            if clip:  # check if the agent can get ammo
                if tracker(clip) == p_coord and get_dist(clip, player) <= 200:
                    state_vec[reload_index] = 2.0
                else:
                    state_vec[reload_index] = 1.0

            while True:
                step += 1
                reward = -1
                p_coord = tracker(player)
                act = self.lstrat.choose_action(v_wrap(state_vec[None, :]))
                r_act, m_act, c_act, n_act, override = self.helm(nav_vec, state, combat, patrol_targ, clip, med, act,
                                                                 targ_coord, tir, p_coord, player, ammo)

                if override:
                    over_ride = True

                elif tracker2(player) != 6:
                    over_ride = False

                if act == 0:  # combat
                    my_act = actions[c_act]

                    if c_act == 7:
                        reward = -2
                    else:
                        if c_act == 4 and step < 500:
                            reward += 1
                        reward += 1
                        over_ride = False
                        rotate_count = 0

                else:  # navigation
                    immobile = False
                    if over_ride and tracker2(player) == 6:  # override for moving through door
                        my_act = "forward"
                    elif act == 1:  # navigation default skill
                        my_act = actions2[n_act]

                    elif act == 2:  # health skill
                        if m_act < 7:
                            my_act = actions2[m_act]
                        else:

                            my_act = 'nothing'
                            immobile = True
                            reward = -2
                    else:  # ammo skill
                        if r_act < 7:
                            my_act = actions2[r_act]
                        else:
                            my_act = 'nothing'
                            immobile = True
                            reward = -2
                    if not immobile and stuck:  # helps the agent get unstuck

                        act_temp = prev_act
                        while act_temp == prev_act:
                            act_temp = actions2[np.random.randint(0, 4)]

                            my_act = act_temp

                        stuck = False

                # count number of times the agent turns
                if my_act == 'turn_left' or my_act == 'turn_right':
                    rotate_count += 1
                else:
                    rotate_count = 0

                if rotate_count >= 10:  # if the agent starts rotating to much mark it as stuck
                    stuck = True

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # make action
                new_state, performance, done, victory, dead = self.game.step(my_act)

                # get new values
                player = new_state['player']

                pl_x2 = player['x_position']
                pl_y2 = player['y_position']

                n_health = int(player['health'])
                n_ammo = int(player['ammo'])
                nstate_vec, nnav_vec, e_temp, combat, clip, med, targ_coord, tir, can_kill = breaker(new_state, pillar)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # change patrol_targ to next target in list when agent gets close this causes the agent to move in a rough circle in the central room
                if get_dist(player, self.patrol_list[patrol_targ]) <= 60.0:
                    patrol_targ += 1
                    if patrol_targ == 4:
                        patrol_targ = 0

                # check if player is stuck
                if int(pl_x) == int(pl_x2) and int(pl_y) == int(pl_y2):
                    if my_act == 'left' or my_act == 'right' or my_act == 'backward' or my_act == 'forward':
                        stuck = True
                        prev_act = my_act

                if victory:  # check for victory
                    victory_count += 1
                    if step < 751:
                        reward += 10

                pl_x = pl_x2
                pl_y = pl_y2

                if e_temp < e_count:  # check for kills
                    reward += 100

                    kills += 1
                    if step < 100:
                        reward += 20
                    elif step < 400:
                        reward += 10

                if n_ammo > ammo:  # check for new ammo
                    reward += 10
                    if act == 3:
                        reward += 50

                    a_count += 1

                if n_health > health:  # check for more health

                    reward += 10
                    if act == 2:
                        reward += 50
                    h_count += 1

                elif n_health < health:  # check for damage
                    reward -= 1

                e_count = e_temp
                ammo = n_ammo
                health = n_health

                nstate_vec[task_index] = task_var

                if e_count > 0:
                    a_check = can_kill
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
                            self.info_list.put([episode, ep_reward, step, performance, kills, a_count, h_count])

                            episode += 1

                            pref_total += performance
                            task = "task4"
                            if TASK_5:
                                task = "task5"
                            print(
                                "Ep:", episode, "enemies:", t_count, "kills:", kills, "victory:", victory,
                                "dead:", dead, "ammo:", a_count,
                                "health:", h_count,
                                "| Ep_r: %.2f" % (pref_total / episode), " indiv: %.2f" % performance, task
                            )
                        else:
                            self.info_list.put([self.g_ep.value, ep_reward, step, performance, kills, a_count, h_count])

                            record_fell(self.g_ep, self.g_ep_r, performance, self.res_queue, self.name, t_count, kills,
                                        victory,
                                        dead, a_count, h_count, task_var, self.my_jump, self.my_asym)
                        break
                state_vec = nstate_vec
                state = new_state
                nav_vec = nnav_vec
                total_step += 1
        if IS_TEST:
            self.test_results.put([victory_count, np.average(raw_list), np.average(pref_list)])
        self.my_jump.put(None)
        self.my_asym.put(None)
        self.info_list.put(None)
        self.res_queue.put(None)

def train_agent(base_file, test_results, my_res, new_file, train_metrics, nav_room, nav_object, raw_file, cp_count):
    gstrat = nav(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, H_SIZE)  # global network

    my_jump = mp.Queue()  #
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

    workers = [
        Worker(gstrat, nav_room, nav_object, opt, global_ep, global_ep_r, res_queue, i, act_net, stric, test_results, my_jump, my_asym,
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
        temp = train_metrics
        temp.append([m_jump, m_asym])

        return my_res2, temp
    return my_res, train_metrics


if __name__ == "__main__":
    n = len(sys.argv)
    control = "Y"
    isa2c = "N"
    t_dir = "4"
    if n == 4:
        control = sys.argv[1]
        isa2c = sys.argv[2]
        t_dir = sys.argv[3]
    else:
        print("invalid arguments need control, is_a2c")

    x = 0
    rang = 2
    test_ep = 10
    nav_room = nav(13, 6)
    nav_item = nav(13, 6)

    nav_room.load_state_dict(torch.load("nav_room.txt"))
    nav_item.load_state_dict(torch.load("nav_item.txt"))
    is_load = "N"  # input("continue Y or N:")
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

    fname = "boot_"
    if IS_CONTROL:
        fname = "control_"
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

        my_res, train_metrics = train_agent(base_file, test_results, my_res, new_file, train_metrics, nav_room, nav_item, raw_file, cp_count)

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

        _, _ = train_agent(base_file, test_results, my_res, new_file, train_metrics, nav_room, nav_item, raw_file, cp_count)  # my_jump, my_asym, my_tran)
    # name of csv file
    filename = "boot_" + tdir + "_a2c.csv"
    outname = "boot_"+ tdir +"_a2c.txt"
    first_line = "boot\n"
    if IS_CONTROL:
        filename = "control_" + tdir + "_a2c.csv"
        outname = "control_" + tdir + "_a2c.txt"
        first_line = "control\n"
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
