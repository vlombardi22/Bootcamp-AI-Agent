import torch
from doom_util import v_wrap, tracker, get_dist, \
    in_center2, in_center3, tracker2, to_center, to_border, breaker, record_fell_ppo
import torch.multiprocessing as mp
from Nav import Net as nav
from ppo_util import Agent

import numpy as np
from viz_task45 import SailonViz as SViz

import random
import csv

UPDATE_GLOBAL_ITER = 20
MAX_EP = 100

IS_TEST = False

STATE_SIZE = 25
ACTION_SIZE = 4


def helper(player, combat, state):
    check = True
    if not in_center2(player) and combat == 0:
        obst = state['items']['obstacle']
        for o in obst:
            if get_dist(o, player) < 80:
                check = False
    return check


class Worker():

    def __init__(self, gnet, gnav, gammo, global_ep, global_ep_r, res_queue, my_queue, p_queue, my_p2, info_list):


        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet = gnet
        self.gnav = gnav
        self.gammo = gammo
        self.my_q = my_queue
        self.info_list = info_list
        self.my_ju = p_queue
        self.my_as = my_p2

        self.c_list = [{"x_position": 0.0, "y_position": 0.0}, {"x_position": 0.0, "y_position": 458.0},
                       {"x_position": 0.0, "y_position": -458.0}, {"x_position": 458.0, "y_position": 0.0},
                       {"x_position": -458.0, "y_position": 0.0},
                       {"x_position": 0.0, "y_position": 0.0}]
        self.s_list = [{"x_position": 180.0, "y_position": 0.0}, {"x_position": 0.0, "y_position": 180.0},
                       {"x_position": -180.0, "y_position": 0}, {"x_position": 0, "y_position": -180.0}]

        seed = 97

        self.seed_list = []
        self.use_seed = False

        if IS_TEST:
            self.use_seed = True
            random.seed(seed)
            np.random.seed(seed)

            self.seed_list = [np.random.randint(0, 1000) for i in range(MAX_EP)]

        use_mock = 0
        use_novel = 1
        level = 208
        difficulty = 'easy'

        self.game = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed)


    def helm(self, nav_vec, state, combat, i_targ, clip, med, act, targ_coord, tir, p_coord, player, ammo):
        """

        :param nav_vec: sensor vector for navigation agents
        :param state: current environment state
        :param combat:
        :param i_targ: search pattern coods
        :param clip: most optimal ammo pack
        :param med: most optimal health pack
        :param act: objective picked by skillnet
        :param targ_coord:
        :param tir: target in room is the target in the current room
        :param p_coord: the room that the player is currently in
        :param player: player object
        :param ammo: ammo
        :return:
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
        if act == 3 and clip:
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

            if tir:

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

        i3 = STATE_SIZE - 1
        i4 = STATE_SIZE - 3
        i5 = STATE_SIZE - 2
        i6 = STATE_SIZE - 4
        task_var = 0.0
        r_list = np.zeros([MAX_EP])  # []

        r_list2 = np.zeros([MAX_EP])  # []
        seed = 0
        episode = 0

        my_av = 0.0

        while (not IS_TEST and self.g_ep.value < MAX_EP) or (IS_TEST and episode < MAX_EP):

            if IS_TEST:
                seed = self.seed_list[episode]

            state = self.game.reset(seed)
            test_obst = state['items']['obstacle']

            state_vec, nav_vec, e_count, combat, clip, med, targ_coord, tir, seek = breaker(state,
                                                                                            test_obst)  # initial state_vec

            step = 0
            kills = 0
            i_targ = 0
            a_count = 0
            h_count = 0
            pact = 'nothing'
            stuck = False
            rotate_count = 0
            ep_rr = 0.0
            over_ride = False
            t_count = e_count

            player = state['player']
            hp = int(player['health'])
            pl_x = player['x_position']
            pl_y = player['y_position']
            ammo = int(player['ammo'])
            p_coord = tracker(player)

            state_vec[i3] = task_var

            if e_count > 0:  # can attack
                a_check = seek
                if combat > 0:
                    if a_check:
                        state_vec[i4] = 3.0
                    else:
                        state_vec[i4] = 2.0
                elif a_check:
                    state_vec[i4] = 1.0

            if med and ACTION_SIZE == 4:
                if tracker(med) == p_coord and get_dist(med, player) <= 200:
                    state_vec[i6] = 2.0
                else:
                    state_vec[i6] = 1.0

            if clip:  # r_act < 7:
                if tracker(clip) == p_coord and get_dist(clip, player) <= 200:
                    state_vec[i5] = 2.0
                else:
                    state_vec[i5] = 1.0

            while True:
                step += 1
                reward = -1
                p_coord = tracker(player)
                act, prob, val = self.gnet.choose_action(state_vec)

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

                            my_act = actions2[m_act]

                        else:

                            my_act = 'nothing'
                            immobile = True
                            reward = -2
                    else:
                        # print("ammo")
                        if r_act < 7:
                            my_act = actions2[r_act]

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

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                new_state, performance, done, victory, dead = self.game.step(my_act)

                player = new_state['player']

                pl_x2 = player['x_position']
                pl_y2 = player['y_position']

                health = int(player['health'])
                n_ammo = int(player['ammo'])
                nstate_vec, nnav_vec, e_temp, combat, clip, med, targ_coord, tir, seek = breaker(new_state, test_obst)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if get_dist(player, self.s_list[i_targ]) <= 60.0:
                    i_targ += 1
                    if i_targ == 4:
                        i_targ = 0

                if int(pl_x) == int(pl_x2) and int(pl_y) == int(pl_y2):
                    if my_act == 'left' or my_act == 'right' or my_act == 'backward' or my_act == 'forward':
                        stuck = True
                        pact = my_act

                if victory:
                    v_count += 1
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
                    reward += 10
                    if act == 3:
                        reward += 50

                    a_count += 1

                if health > hp:

                    reward += 10
                    if act == 2:
                        reward += 50
                    h_count += 1

                elif health < hp:
                    reward -= 1

                ammo = n_ammo
                hp = health

                nstate_vec[i3] = task_var

                if e_count > 0:
                    a_check = seek
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

                    self.gnet.remember(state_vec, act, prob, val, reward, done)
                if (
                        not IS_TEST and total_step % UPDATE_GLOBAL_ITER == 0) or done:  # update global and assign to local net
                    # sync
                    if not IS_TEST:
                        self.gnet.learn()

                    if done:  # done and print information

                        if IS_TEST:
                            r_list[episode] = performance
                            r_list2[episode] = ep_rr
                            self.info_list.put([episode, ep_rr, step, performance, kills, a_count, h_count])

                            episode += 1

                            my_av += performance

                            print(
                                "Ep:", episode, "enemies:", t_count, "kills:", kills, "victory:", victory,
                                "dead:", dead, "ammo:", a_count,
                                "health:", h_count,
                                "| Ep_r: %.2f" % (my_av / episode), " indiv: %.2f" % performance, "task5"
                            )
                        else:
                            self.info_list.put([self.g_ep.value, ep_rr, step, performance, kills, a_count, h_count])
                            record_fell_ppo(self.g_ep, self.g_ep_r, performance, self.res_queue, t_count, kills,
                                        victory,
                                        dead, a_count, h_count, task_var, self.my_ju, self.my_as)

                        break
                state_vec = nstate_vec
                state = new_state
                nav_vec = nnav_vec
                total_step += 1
        if IS_TEST:
            av = np.average(r_list)
            self.my_q.put([v_count, np.average(r_list2), av])
        self.my_ju.put(None)
        self.my_as.put(None)
        self.info_list.put(None)
        self.res_queue.put(None)


def main(f, my_q, my_r, f2, my_q2, f3):
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    myshape = np.zeros(STATE_SIZE)

    gnet = Agent(n_actions=ACTION_SIZE, input_dims=myshape.shape, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
    gnav = nav(13, 6)
    gammo = nav(13, 6)

    gnav.load_state_dict(torch.load("nav_room.txt"))
    gammo.load_state_dict(torch.load("nav_item.txt"))
    my_jump = mp.Queue()
    my_asym = mp.Queue()
    my_info = mp.Queue()

    gnet.load_weights(f)

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    if IS_TEST:
        print("testing")
    else:
        print("training")

    worker = Worker(gnet, gnav, gammo, global_ep, global_ep_r, res_queue, my_q, my_jump, my_asym, my_info)
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
    with open(f3, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        csvwriter.writerows(myinfo)
    csvfile.close()

    if not IS_TEST:
        gnet.save_weights(f2)

        my_r2 = np.add(my_r, res)

        my_q3 = my_q2
        my_q3.append([m_jump, m_asym])
        return my_r2

    return my_r


if __name__ == "__main__":
    x = 0
    rang = 2
    test_ep = 10

    control = "N"  # input("control Y or N:")
    testing = "N"  # input("test Y or N:")
    is_load = "N"  # input("continue Y or N:")
    is_control = False
    if control == "Y" or control == "y":
        is_control = True
    if testing == "Y" or testing == "y":
        IS_TEST = True

    if IS_TEST:
        MAX_EP = test_ep

    my_q = mp.Queue()
    my_q2 = []
    my_r = np.zeros([MAX_EP])
    pref = "ppoboot_"
    if is_control:
        pref = "ppocontrol_"

    for ind in range(rang):
        n = ind + x
        f0 = pref + str(n)
        f1 = f0 + ".txt"
        f2 = pref + "task5_" + str(n) + ".txt"
        f3 = f0 + "raw.csv"

        print(f1)
        my_r = main(f1, my_q, my_r, f2, my_q2, f3)


    IS_TEST = True

    if IS_TEST:
        MAX_EP = test_ep
    my_q = mp.Queue()
    f2 = "dud.txt"
    for ind in range(rang):
        n = ind + x
        f0 = pref + str(n)
        f1 = pref + "task5_" + str(n) + ".txt"
        f3 = f0 + "rawtest.csv"

        print(f1)
        _ = main(f1, my_q, my_r, f2, my_q2, f3)
    # name of csv file
    filename = "boot_ppo_task5.csv"
    outname = "boot_task5_ppo.txt"
    w = "boot\n"
    if is_control:
        outname = "control_task5_ppo.txt"
        filename = "control_ppo_task5.csv"
        w = "control\n"

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

    f = open(outname, "w")

    f.write(w)
    while True:
        r = my_q.get()
        if r is not None:
            print(r)
            mystr = str(r) + "\n"
            f.write(mystr)

        else:
            break
    f.write("other\n")
    print("other")
    for i in my_q2:
        mystr2 = str(i) + "\n"
        f.write(mystr2)
        print(i)
    f.close()
    print("done")
