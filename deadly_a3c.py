"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
from myutil import v_wrap, push_and_pull

import torch.multiprocessing as mp
from hall_skills import get_dist, get_angle, get_armor, fight, record_dead, navigate
from shared import SharedAdam
from Nav import Net as nav

import csv
import numpy as np
import vizdoom as vzd
import random
import os
import sys

os.environ["OMP_NUM_THREADS"] = "4"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.97
MAX_EP = 1000
HIDDEN_SIZE = 32
H_SIZE = 16
IS_CONTROL = False
IS_TEST = False

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


class Worker(mp.Process):

    def __init__(self, gstrat, opt, global_ep, global_ep_r, res_queue, name, f, stric, test_results, my_jump, my_asym,
                 info_list,
                 l):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gstrat, self.opt = gstrat, opt

        self.lstrat = nav(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, H_SIZE)

        self.seed_list = []
        if IS_TEST:
            seed = 97
            random.seed(seed)
            np.random.seed(seed)
            self.seed_list = [np.random.randint(0, 1000) for i in range(MAX_EP)]
        if l == "Y":
            self.lstrat.load_state_dict(f, strict=stric)  # bookmark

        self.test_results = test_results
        self.my_jump = my_jump
        self.my_asym = my_asym

        self.game = vzd.DoomGame()
        self.step_limit = 1500  # 2100
        self.info_list = info_list
        self.game.set_doom_scenario_path("deadly_hall.wad")
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
        task_index = STATE_SIZE - 1  # bookmark
        reload_index = STATE_SIZE - 2
        combat_index = STATE_SIZE - 3
        heal_index = STATE_SIZE - 4

        ep = 0
        task_var = 0.0
        pref_list = []
        raw_list = []
        t_kills = 0

        while self.g_ep.value < MAX_EP:
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
            state_vec[heal_index] = 1.0

            ep_reward = 0.0
            ep += 1
            tk = 0
            fired = 0

            while True:
                step += 1
                reward = -1

                act = self.lstrat.choose_action(v_wrap(state_vec[None, :]))

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

                    elif act == 1:  # ammo

                        if r_act < 7:
                            my_act = actions[r_act]
                        else:

                            reward -= 2
                            my_act = actions[7]
                    else:  # armor
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
                    nstate_vec[heal_index] = 1.0
                    if victory:
                        nstate_vec[heal_index] = 0.0
                    elif dist < 200:
                        nstate_vec[heal_index] = 2.0


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

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync

                    if len(buffer_s) > 0 and not IS_TEST:
                        push_and_pull(self.opt, self.lstrat, self.gstrat, done, nstate_vec, buffer_s, buffer_a, buffer_r,
                                      GAMMA)

                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        t_kills += kills
                        if IS_TEST:
                            pref_list.append(performance)
                            raw_list.append(ep_reward)
                        self.info_list.put([self.g_ep.value, ep_reward, step, performance, kills, a_count])
                        record_dead(self.g_ep, self.g_ep_r, performance, self.res_queue, self.name, t_count, kills, victory,
                                    dead, a_count, self.my_jump, o_dist, step, tk, self.my_asym)
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


def train_agent(base_file, test_results, my_res, new_file, my_res2, raw_file, cp_count):
    gstrat = nav(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, H_SIZE)  # global network

    my_jump = mp.Queue()
    my_asym = mp.Queue()
    my_info = mp.Queue()
    l = "Y"
    stric = False
    if IS_TEST:
        l = "Y"
        stric = True
    act_net = {}

    if l == "Y":

        act_full = torch.load(base_file)

        if not stric:  # critic layers 1-3 work kind of

            act_net['actor1.weight'] = act_full['actor1.weight']
            act_net['actor1.bias'] = act_full['actor1.bias']
            act_net['critic1.weight'] = act_full['critic1.weight']
            act_net['critic1.bias'] = act_full['critic1.bias']
            act_net['actor2.weight'] = act_full['actor2.weight']
            act_net['actor2.bias'] = act_full['actor2.bias']
            act_net['critic2.weight'] = act_full['critic2.weight']
            act_net['critic2.bias'] = act_full['critic2.bias']
            act_net['actor3.weight'] = act_full['actor3.weight']
            act_net['actor3.bias'] = act_full['actor3.bias']
            act_net['critic3.weight'] = act_full['critic3.weight']
            act_net['critic3.bias'] = act_full['critic3.bias']

        else:

            act_net = act_full

        gstrat.load_state_dict(act_net, strict=stric)

    my_lr = 1e-3
    if IS_TEST:
        my_lr = 1e-4  # TODO ????????????????????
    opt = SharedAdam(gstrat.parameters(), lr=my_lr, betas=(0.90, 0.999))  # global optimizer
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
        Worker(gstrat, opt, global_ep, global_ep_r, res_queue, i, act_net, stric, test_results, my_jump, my_asym, my_info, l) for
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
        print(new_file)
        torch.save(gstrat.state_dict(), new_file)

        my_r2 = np.add(my_res, res)
        my_q3 = my_res2
        my_q3.append([m_jump, m_asym])

        return my_r2, my_q3
    return my_res, my_res2


if __name__ == "__main__":
    n = len(sys.argv)
    isa2c = "N"
    control = "N"

    if n == 3:
        control = sys.argv[1]
        isa2c = sys.argv[2]
    else:
        print("invalid arguments need control, is_a2c")
    x = 0
    rang = 15
    test_ep = 1000

    is_load = "N"

    if control == "Y" or control == "y":
        IS_CONTROL = True

    test_results = mp.Queue()
    train_metrics = []
    my_res = np.zeros([MAX_EP])
    fname = "boot_"
    if IS_CONTROL:
        fname = "control_"
    is_a2c = False
    if isa2c == "Y":
        is_a2c = True
    cp_count = 4
    if is_a2c:
        fname = fname + "a2c_"
        cp_count = 1


    for ind in range(rang):
        n = ind + x
        f_temp = fname + str(n)
        base_file = "tasks123/" + f_temp + ".txt"
        new_file = "task6/" + fname + "deadly_" + str(n) + ".txt"
        raw_file = "task6/" + f_temp + "raw.csv"
        if not os.path.exists(base_file):
            print("file:", base_file, "does not exist")
            break
        my_res, train_metrics = train_agent(base_file, test_results, my_res, new_file, train_metrics, raw_file)

    IS_TEST = True

    MAX_EP = test_ep

    test_results = mp.Queue()
    new_file = "dud.txt"
    cp_count = 1
    for ind in range(rang):
        n = ind + x
        f_temp = fname + str(n)
        base_file = "task6/" + fname + "deadly_" + str(n) + ".txt"
        raw_file = "task6/" + f_temp + "rawtest.csv"
        if not os.path.exists(base_file):
            print("file:", base_file, "does not exist")
            break
        _, _ = train_agent(base_file, test_results, my_res, new_file, train_metrics, raw_file, cp_count)
    # name of csv file
    filename = "boot_deadly.csv"
    outname = "boot_deadly.txt"
    if IS_CONTROL:
        filename = "control_deadly.csv"
        outname = "control_deadly.txt"
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
    if control == "N":
        f.write("boot\n")
    else:
        f.write("control\n")
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
