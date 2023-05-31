"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch.multiprocessing as mp
from boot_utils.hall_skills import get_dist, get_angle, get_armor, fight, record_fell, navigate
from boot_utils.ppo_util import Agent
import csv
import numpy as np
import vizdoom as vzd
import random
import os
import sys
os.environ["OMP_NUM_THREADS"] = "4"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.97
MAX_EP = 10
HIDDEN_SIZE = 32
H_SIZE = 16
IS_CONTROL = False
IS_TEST = False
DEBUG = False

STATE_SIZE = 25
ACTION_SIZE = 3



def break_armor(armor, player):
    if not armor:
        return [0.0, 0.0, -1.0, 0.0], 0.0

    angle, _ = get_angle(armor, player, 0.0)
    angle = angle * 180 / np.pi
    dist = get_dist(player, armor)
    strat_armor = [armor.position_x, armor.position_y, dist, angle]

    return strat_armor, dist


def break_item(item, player):
    if len(item) == 0:
        return [0.0, 0.0, -1.0, 0.0], None
    min_dist = 100000
    m_item = None
    for a in item:
        dist = get_dist(player, a)
        if min_dist > dist:
            min_dist = dist
            m_item = a
    angle, _ = get_angle(m_item, player, 0.0)
    angle = angle * 180 / np.pi
    strat_ammo = [m_item.position_x, m_item.position_y, get_dist(m_item, player), angle]
    return strat_ammo, m_item


def break_enemy(enemies, player):
    strat_enemy = []
    min_dist = 10000
    m_enemy = None
    for e in enemies:
        dist = get_dist(player, e)
        if min_dist > dist:
            min_dist = dist
            m_enemy = e
    if not m_enemy:
        strat_enemy = [0.0, 0.0, 0.0, -1.0, 0.0]
    else:

        angle, _ = get_angle(m_enemy, player, 0.0)
        angle = angle * 180 / np.pi
        e_type = 1
        if m_enemy.name == "ShotgunGuy":
            e_type = 2
        elif m_enemy.name == "ChaingunGuy":
            e_type = 3
        strat_enemy = [m_enemy.position_x, m_enemy.position_y, e_type, get_dist(m_enemy, player),
                       angle]
        if min_dist > 250.0:
            m_enemy = None
    return m_enemy, strat_enemy

class Worker():

    def __init__(self, strategist, global_ep, global_ep_r, res_queue, name, test_results, my_jump, my_asym, info_list):

        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.strategist = strategist

        self.test_results = test_results
        self.info_list = info_list
        self.my_jump = my_jump
        self.my_asym = my_asym



        self.seed_list = []
        if IS_TEST:
            seed = 97
            random.seed(seed)
            np.random.seed(seed)
            self.seed_list = [np.random.randint(0, 1000) for i in range(MAX_EP)]

        self.game = vzd.DoomGame()
        self.step_limit = 50#1500

        self.game.set_doom_scenario_path("scenarios/deadly_hall.wad")
        # Sets map to start (scenario .wad files can contain many maps).
        self.game.set_doom_map("map01")

        # Sets resolution. Default is 320X240
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

        # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        # Enables depth buffer.
        self.game.set_depth_buffer_enabled(True)

        # Enables labeling of in game objects labeling.
        self.game.set_labels_buffer_enabled(True)

        # Enables buffer with top down map of the current episode/level.
        self.game.set_automap_buffer_enabled(True)

        # Enables information about all objects present in the current episode/level.
        self.game.set_objects_info_enabled(True)

        # Enables information about all sectors (map layout).
        self.game.set_sectors_info_enabled(True)

        # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
        self.game.set_render_hud(True)
        self.game.set_render_minimal_hud(False)  # If hud is enabled
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)  # Bullet holes and blood on the walls
        self.game.set_render_particles(False)
        self.game.set_render_effects_sprites(False)  # Smoke and blood
        self.game.set_render_messages(False)  # In-game messages
        self.game.set_render_corpses(False)
        self.game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

        # Adds buttons that will be allowed.

        self.game.add_available_button(vzd.Button.MOVE_LEFT)
        self.game.add_available_button(vzd.Button.MOVE_RIGHT)
        self.game.add_available_button(vzd.Button.ATTACK)
        self.game.add_available_button(vzd.Button.MOVE_FORWARD)
        self.game.add_available_button(vzd.Button.MOVE_BACKWARD)
        self.game.add_available_button(vzd.Button.TURN_LEFT)
        self.game.add_available_button(vzd.Button.TURN_RIGHT)

        # Adds game variables that will be included in state.
        self.game.add_available_game_variable(vzd.GameVariable.HEALTH)
        self.game.add_available_game_variable(vzd.GameVariable.AMMO2)
        # Causes episodes to finish after 200 tics (actions)
        self.game.set_episode_timeout(self.step_limit)

        # Makes episodes start after 10 tics (~after raising the weapon)
        self.game.set_episode_start_time(10)

        # Makes the window appear (turned on by default)
        self.game.set_window_visible(False)

        # Sets the living (for each move) to -1
        # self.game.set_living_reward(-1)
        self.game.set_death_penalty(50)
        self.total_target_count = 0
        self.max_target_count = 8

        # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
        self.game.set_mode(vzd.Mode.PLAYER)

        # Initialize the game. Further configuration won't take any effect from now on.
        self.game.init()

    def breaker(self, state):
        objects = state.objects
        player = None
        armor = None
        enemies = []
        items = []
        for o in objects:

            if o.name == "DoomPlayer":
                player = o
            elif o.name == "GreenArmor":
                armor = o
            elif o.name == "ChaingunGuy" or o.name == "ShotgunGuy" or o.name == "Zombieman":
                enemies.append(o)
            elif o.name == "Clip":
                items.append(o)

        target, strat_enemy = break_enemy(enemies, player)

        e_count = len(enemies)
        a_count = 0
        if armor:
            a_count = 1
        i_count = len(items)

        strat_armor, dist = break_armor(armor, player)

        ammo = self.game.get_game_variable(vzd.GameVariable.AMMO2)

        strat_item, clip = break_item(items, player)
        health = self.game.get_game_variable(vzd.GameVariable.HEALTH)

        sensor_vec = [player.position_x, player.position_y, player.angle, float(ammo), float(health),
                      e_count] + strat_enemy + [
                         i_count] + strat_item + [a_count] + strat_armor + [0.0, 0.0, 0.0, 0.0]

        r_act = 7
        m_act = 7
        t = [0, 1, 3, 4]
        n_act = t[np.random.randint(0, 4)]
        c_act = 7

        if clip:
            r_act = get_armor(clip, player)

        if armor:
            m_act = navigate(armor, player)
        if target:
            c_act = fight(target, player)

        return np.asarray(sensor_vec), e_count, r_act, c_act, n_act, m_act, health, ammo, dist

    def run(self):

        total_step = 1
        actions = [[True, False, False, False, False, False, False], [False, True, False, False, False, False, False],
                   [False, False, True, False, False, False, False], [False, False, False, True, False, False, False],
                   [False, False, False, False, True, False, False], [False, False, False, False, False, True, False],
                   [False, False, False, False, False, False, True], [False, False, False, False, False, False, False]]
        v_count = 0
        task_index = STATE_SIZE - 1
        combat_index = STATE_SIZE - 3
        reload_index = STATE_SIZE - 2
        health_index = STATE_SIZE - 4
        ep = 0
        task_var = 0.0
        pref_list = []
        raw_list = []
        t_kills = 0

        my_av = 0.0
        while (not IS_TEST and self.g_ep.value < MAX_EP) or (IS_TEST and ep < MAX_EP):
            step = 0

            if IS_TEST:
                seed = self.seed_list[self.g_ep.value]
                self.game.set_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
            self.game.new_episode()

            state = self.game.get_state()

            kills = 0
            self.total_target_count = 0
            state_vec, e_count, r_act, c_act, n_act, m_act, health, ammo, o_dist = self.breaker(
                state)  # initial state_vec

            t_count = e_count

            buffer_s, buffer_a, buffer_r = [], [], []

            a_count = 0


            state_vec[task_index] = task_var

            if e_count > 0:  # can attack

                if c_act < 7:
                    state_vec[combat_index] = 3.0
                else:
                    state_vec[combat_index] = 1.0

            state_vec[reload_index] = 0.0
            if r_act < 7:
                state_vec[reload_index] = 2.0
            state_vec[health_index] = 1.0

            ep_reward = 0.0
            ep += 1
            tk = 0
            fired = 0

            while True:
                step += 1
                reward = -1

                act, prob, val = self.strategist.choose_action(state_vec)


                if act == 0:
                    my_act = actions[c_act]
                    if c_act == 7:
                        reward -= 2
                    else:
                        reward += 1
                else:
                    if act == 3:
                        my_act = actions[n_act]
                        if e_count == 0 or c_act < 7:
                            reward -= 1

                    elif act == 1: # ammo

                        if r_act < 7:
                            my_act = actions[r_act]
                        else:

                            reward -= 2

                            my_act = actions[7]
                    else:
                        if m_act < 7:
                            my_act = actions[m_act]
                        else:
                            reward -= 2

                            my_act = actions[7]





                rew = self.game.make_action(my_act, 1)
                victory = False
                new_state = self.game.get_state()
                done = self.game.is_episode_finished()
                dead = self.game.is_player_dead()

                if rew > 1:
                    victory = True
                elif done and not dead:
                    reward -= 5

                reward += rew

                nstate_vec = []
                if not done:

                    pact = c_act
                    nstate_vec, e_temp, r_act, c_act, n_act, m_act, h, n_ammo, dist = self.breaker(new_state)


                    if dist < o_dist and act > 0:
                        reward += 0.5

                    o_dist = dist
                    if n_ammo < ammo:
                        fired += 1
                    if e_temp < e_count:
                        if act == 0 and pact < 7:
                            reward += 15
                            kills += 1
                            tk += 1

                            fired = 0
                        else:
                            tk += 1
                    e_count = e_temp

                    if h < health:
                        reward -= 1.0

                    health = h

                    if n_ammo > ammo:
                        reward += 10

                    ammo = n_ammo

                    nstate_vec[task_index] = task_var
                    nstate_vec[combat_index] = 0.0
                    if e_count > 0:
                        if c_act < 7:

                            nstate_vec[combat_index] = 3.0

                        else:
                            nstate_vec[combat_index] = 1.0

                    nstate_vec[reload_index] = 0.0
                    if r_act < 7:
                        nstate_vec[reload_index] = 2.0
                    nstate_vec[health_index] = 1.0
                    if victory:
                        nstate_vec[health_index] = 0.0
                    elif dist < 200:
                        nstate_vec[health_index] = 2.0


                else:
                    nstate_vec = state_vec
                    if dead:
                        nstate_vec[4] = 0
                arm = 2
                if victory:
                    v_count += 1
                    done = True

                    arm = 0
                current_targets = 0
                current_targets = current_targets + (t_count - kills) + arm
                self.total_target_count = self.total_target_count + current_targets
                target_by_time = current_targets * (self.step_limit - step)
                performance = 1 - (self.total_target_count + target_by_time) / (
                        self.step_limit * self.max_target_count)
                performance = round(performance, 6)

                ep_reward += reward



                if not IS_TEST:
                    buffer_a.append(act)
                    buffer_s.append(state_vec)
                    buffer_r.append(reward)
                    self.strategist.remember(state_vec, act, prob, val, reward, done)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync

                    if not IS_TEST:
                        self.strategist.learn()

                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        t_kills += kills
                        if IS_TEST:
                            pref_list.append(performance)
                            raw_list.append(ep_reward)
                            self.info_list.put([ep, ep_reward, step, performance, kills, a_count])
                            my_av += performance
                            print(
                                self.name,
                                "Ep:", ep, "enemies:", t_count, "kills:", kills, "victory:", victory,
                                "dead:", dead, "ammo:", a_count,
                                "| Ep_r: %.2f" % (my_av / ep), " indiv: %.2f" % performance, "task6")
                        else:

                            self.info_list.put([self.g_ep.value, ep_reward, step, performance, kills, a_count])
                            record_fell(self.g_ep, self.g_ep_r, performance, self.res_queue, self.name, t_count, kills,
                                        victory,
                                        dead, a_count, task_var, self.my_jump, self.my_asym)

                        break

                state_vec = nstate_vec
                state = new_state
                total_step += 1

        if IS_TEST:
            self.test_results.put([v_count, np.average(raw_list), np.average(pref_list)])
        self.my_jump.put(None)
        self.my_asym.put(None)
        self.info_list.put(None)
        self.res_queue.put(None)


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


    stric = False
    if IS_TEST:
        print("testing")
        stric = True
    else:
        print("training")
    strategist.load_weights(base_file, bdir, stric)

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    worker = Worker(strategist, global_ep, global_ep_r, res_queue, 0,test_results, my_jump, my_asym,
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
        # TODO possibly return temp for consistency with fell_a3c.py although both ways work
        return my_res2

    return my_res


if __name__ == "__main__":

    n = len(sys.argv)
    control = "Y"
    if n == 2:
        control = sys.argv[1]
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
    fname = "ppoboot_"
    if is_control:
        fname = "ppocontrol_"
    tdir = "task6"

    fname2 = tdir + "/" + fname

    for ind in range(num_agents):
        n = ind + starting_index

        f_temp = fname + "task123_" + str(n)
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
    filename = "boot_task6_ppo.csv"
    outname = "boot_task6_ppo.txt"
    first_line = "boot\n"
    if is_control:
        filename = "control_task6_ppo.csv"
        outname = "control_task_6_ppo.txt"
        first_line = "control\n"
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
