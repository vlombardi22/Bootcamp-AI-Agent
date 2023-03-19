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
    """
    Determines if a player is close to an obstacle
    :param player: player object
    :param combat: combat object
    :param state: environment state
    :return: true if not close to an obstacle
    """
    check = True
    if not in_center2(player) and combat == 0:
        obst = state['items']['obstacle']
        for o in obst:
            if get_dist(o, player) < 80:
                check = False
    return check


class Worker():

    def __init__(self, strategist, nav_room, nav_object, global_ep, global_ep_r, res_queue, test_results, my_jump, my_asym, info_list):
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.strategist = strategist
        self.nav_room = nav_room
        self.nav_object = nav_object
        self.test_results = test_results
        self.info_list = info_list
        self.my_jump = my_jump
        self.my_asym = my_asym

        # target coordinates for each room
        self.room_list = [{"x_position": 0.0, "y_position": 0.0}, {"x_position": 0.0, "y_position": 458.0},
                          {"x_position": 0.0, "y_position": -458.0}, {"x_position": 458.0, "y_position": 0.0},
                          {"x_position": -458.0, "y_position": 0.0},
                          {"x_position": 0.0, "y_position": 0.0}]
        # coordinates for moving around the main room
        self.patrol_list = [{"x_position": 180.0, "y_position": 0.0}, {"x_position": 0.0, "y_position": 180.0},
                            {"x_position": -180.0, "y_position": 0}, {"x_position": 0, "y_position": -180.0}]

        seed = 97
        use_mock = 0
        use_novel = 1
        level = 208
        difficulty = 'easy'

        self.seed_list = []
        self.use_seed = False

        if IS_TEST:
            self.use_seed = True
            random.seed(seed)
            np.random.seed(seed)

            self.seed_list = [np.random.randint(0, 1000) for i in range(MAX_EP)]

        self.game = SViz(use_mock, use_novel, level, False, seed, difficulty, use_seed=self.use_seed)


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
                    m_act = self.nav_object.choose_action(v_wrap(nav_vec[None, :]))
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
                    r_act = self.nav_object.choose_action(v_wrap(nav_vec[None, :]))

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

                    n_act = self.nav_object.choose_action(v_wrap(nav_vec[None, :]))

                else:
                    nav_vec[4] = self.room_list[p_coord - 1]['x_position']
                    nav_vec[5] = self.room_list[p_coord - 1]['y_position']
                    nav_vec[6] = get_dist(player, self.room_list[p_coord - 1])
                    n_act = self.nav_object.choose_action(v_wrap(nav_vec[None, :]))

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
            test_obst = state['items']['obstacle']

            state_vec, nav_vec, e_count, combat, clip, med, targ_coord, tir, seek = breaker(state,
                                                                                            test_obst)  # initial state_vec

            step = 0
            kills = 0
            patrol_targ = 0  # coordinate for moving in a rough circle around the central room
            a_count = 0
            h_count = 0
            prev_act = 'nothing'
            stuck = False
            rotate_count = 0
            ep_reward = 0.0
            over_ride = False
            t_count = e_count

            player = state['player']
            hp = int(player['health'])
            pl_x = player['x_position']
            pl_y = player['y_position']
            ammo = int(player['ammo'])
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
                    if over_ride and tracker2(player) == 6: # override for moving through door
                        my_act = "forward"
                    elif act == 1:  # navigation
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
                    if not immobile and stuck: # helps the agent get unstuck

                        act_temp = prev_act
                        while act_temp == prev_act:
                            act_temp = actions2[np.random.randint(0, 4)]

                            my_act = act_temp

                        stuck = False

                if my_act == 'turn_left' or my_act == 'turn_right':
                    rotate_count += 1
                else:
                    rotate_count = 0

                if rotate_count >= 10: # if the agent starts rotating to much mark it as stuck
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
                # change patrol_targ to next target in list when agent gets close this causes the agent to move in a rough circle in the central room
                if get_dist(player, self.patrol_list[patrol_targ]) <= 60.0:
                    patrol_targ += 1
                    if patrol_targ == 4:
                        patrol_targ = 0

                if int(pl_x) == int(pl_x2) and int(pl_y) == int(pl_y2):
                    if my_act == 'left' or my_act == 'right' or my_act == 'backward' or my_act == 'forward':
                        stuck = True
                        prev_act = my_act

                if victory:
                    victory_count += 1
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

                            episode += 1

                            pref_total += performance

                            print(
                                "Ep:", episode, "enemies:", t_count, "kills:", kills, "victory:", victory,
                                "dead:", dead, "ammo:", a_count,
                                "health:", h_count,
                                "| Ep_r: %.2f" % (pref_total / episode), " indiv: %.2f" % performance, "task5"
                            )
                        else:
                            self.info_list.put([self.g_ep.value, ep_reward, step, performance, kills, a_count, h_count])
                            record_fell_ppo(self.g_ep, self.g_ep_r, performance, self.res_queue, t_count, kills,
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


def main(base_file, test_results, my_res, new_file, train_metrics, raw_file):
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    myshape = np.zeros(STATE_SIZE)

    strategist = Agent(n_actions=ACTION_SIZE, input_dims=myshape.shape, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
    nav_room = nav(13, 6)
    nav_object = nav(13, 6)

    nav_room.load_state_dict(torch.load("nav_room.txt"))
    nav_object.load_state_dict(torch.load("nav_item.txt"))
    my_jump = mp.Queue()
    my_asym = mp.Queue()
    my_info = mp.Queue()

    strategist.load_weights(base_file)

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    if IS_TEST:
        print("testing")
    else:
        print("training")

    worker = Worker(strategist, nav_room, nav_object, global_ep, global_ep_r, res_queue, test_results, my_jump, my_asym, my_info)
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
        strategist.save_weights(new_file)

        my_res2 = np.add(my_res, res)

        temp = train_metrics
        temp.append([m_jump, m_asym])
        return my_res2

    return my_res


if __name__ == "__main__":
    starting_index = 0
    num_agents = 2
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

    test_results = mp.Queue()
    my_res = np.zeros([MAX_EP])
    train_metrics = []
    fname = "ppoboot_"
    if is_control:
        fname = "ppocontrol_"

    for ind in range(num_agents):
        n = ind + starting_index
        f_temp = fname + str(n)
        base_file = f_temp + ".txt"
        new_file = fname + "task5_" + str(n) + ".txt"
        raw_file = f_temp + "raw.csv"

        print(base_file)
        my_res = main(base_file, test_results, my_res, new_file, train_metrics, raw_file)

    IS_TEST = True

    if IS_TEST:
        MAX_EP = test_ep
    test_results = mp.Queue()
    new_file = "dud.txt"
    for ind in range(num_agents):
        n = ind + starting_index
        f_temp = fname + str(n)
        base_file = fname + "task5_" + str(n) + ".txt"
        raw_file = f_temp + "rawtest.csv"

        print(base_file)
        _ = main(base_file, test_results, my_res, new_file, train_metrics, raw_file)
    # name of csv file
    filename = "boot_ppo_task5.csv"
    outname = "boot_task5_ppo.txt"
    first_line = "boot\n"
    if is_control:
        outname = "control_task5_ppo.txt"
        filename = "control_ppo_task5.csv"
        first_line = "control\n"

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