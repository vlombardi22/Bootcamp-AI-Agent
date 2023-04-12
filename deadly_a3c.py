"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
import torch.nn as nn
from myutil import v_wrap, set_init, push_and_pull
import torch.nn.functional as F
import torch.multiprocessing as mp
from hall_skills import get_dist, get_angle, get_armor, fight, record_dead, get_ammo, navigate
from shared import SharedAdam
# import cv2
# import matplotlib.pyplot as plt
import csv
import numpy as np
import vizdoom as vzd
import random
import os
from time import sleep

os.environ["OMP_NUM_THREADS"] = "4"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.97
MAX_EP = 1000
HIDDEN_SIZE = 32  # 128
H_SIZE = 16  # 64
IS_CONTROL = False
IS_TEST = False
DEBUG = False

STATE_SIZE = 25  # 12#13  # 15#14#11#8#10#12  #12#8#9#11
ACTION_SIZE = 4


class Net(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.pi1 = nn.Linear(s_dim, HIDDEN_SIZE)
        self.pij = nn.Linear(HIDDEN_SIZE, H_SIZE)
        self.pij2 = nn.Linear(H_SIZE, H_SIZE)
        self.pi2 = nn.Linear(H_SIZE, a_dim)

        self.v1 = nn.Linear(s_dim, HIDDEN_SIZE)
        self.vj = nn.Linear(HIDDEN_SIZE, H_SIZE)
        self.vj2 = nn.Linear(H_SIZE, H_SIZE)

        self.v2 = nn.Linear(H_SIZE, 1)
        set_init([self.pi1, self.pij, self.pij2, self.pi2, self.v1, self.vj, self.vj2, self.v2])
        # set_init([self.pi1, self.pij, self.pi2, self.v1, self.vj, self.v2])

        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = F.leaky_relu(self.pi1(x))
        v1 = F.leaky_relu(self.v1(x))
        pij = F.leaky_relu(self.pij(pi1))
        vj = F.leaky_relu(self.vj(v1))
        pij2 = F.leaky_relu(self.pij2(pij))
        vj2 = F.leaky_relu(self.vj2(vj))
        logits = self.pi2(pij2)
        values = self.v2(vj2)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


def break_armor(armor, player):
    if not armor:
        return [0.0, 0.0, -1.0, 0.0], 0.0

    angle, _ = get_angle(armor, player, 0.0)
    angle = angle * 180 / np.pi
    dist = get_dist(player, armor)
    strat_armor = [armor.position_x, armor.position_y, dist, angle]
    # print(strat_armor)
    # exit()
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
    # o.name == "ChaingunGuy" or o.name == "ShotgunGuy" or o.name == "Zombieman"
    for e in enemies:
        dist = get_dist(player, e)
        # print(e.position_x, ",", e.position_y, ",", e.name)
        # if e.name == "Zombieman":
        #    print(dist)
        """if dist < 250 and min_dist < 250:

            if not m_enemy:
              min_dist = dist
              m_enemy = e  

            elif m_enemy.name == "Zombieman":
                min_dist = dist
                m_enemy = e
            elif m_enemy.name == "ShotgunGuy" and e.name != "Zombieman":
                min_dist = dist
                m_enemy = e
            elif e.name == "ChaingungunGuy":       
                min_dist = dist
                m_enemy = e
        elif min_dist > dist and min_dist >= 250:
        """
        if min_dist > dist:
            min_dist = dist
            m_enemy = e
    # exit()
    if not m_enemy:
        strat_enemy = [0.0, 0.0, 0.0, -1.0, 0.0]
    else:
        # print(m_enemy.name)
        # print(min_dist)
        angle, _ = get_angle(m_enemy, player, 0.0)
        angle = angle * 180 / np.pi
        t = 1
        if m_enemy.name == "ShotgunGuy":
            t = 2
        elif m_enemy.name == "ChaingunGuy":
            t = 3
        # print(m_enemy.health)
        strat_enemy = [m_enemy.position_x, m_enemy.position_y, t, get_dist(m_enemy, player),
                       angle]
        # print(min_dist)
        if min_dist > 250.0:
            # strat_enemy[4] = 180.0

            # strat_enemy = [0.0, 0.0, 0.0, -1.0, 0.0]
            m_enemy = None
    # print(get_dist(m_enemy, player))
    return m_enemy, strat_enemy


class Worker(mp.Process):

    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, f, stric, my_queue, p_queue, my_p2,
                 info_list,
                 l):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt

        self.lnet = Net(STATE_SIZE, ACTION_SIZE)

        self.seed_list = []
        if IS_TEST:
            seed = 97
            random.seed(seed)
            np.random.seed(seed)
            self.seed_list = [np.random.randint(0, 1000) for i in range(MAX_EP)]
        if l == "Y":
            # if IS_TEST:
            #    self.lnet = gnet
            # else:
            # print(f)

            self.lnet.load_state_dict(f, strict=stric)  # bookmark

            # self.lnet.load_state_dict(gnet.state_dict())

            # act_net = {}
            """"
            act_full = gnet.state_dict()
            if not stric:

                # act_net['pi1.weight'] = act_full['pi1.weight']
                # act_net['pi1.bias'] = act_full['pi1.bias']
                # act_net['v1.weight'] = act_full['v1.weight']
                # act_net['v1.bias'] = act_full['v1.bias']
                act_net['pij.weight'] = act_full['pij.weight']
                act_net['pij.bias'] = act_full['pij.bias']
                # act_net['vj.weight'] = act_full['vj.weight']
                # act_net['vj.bias'] = act_full['vj.bias']
                act_net['pij2.weight'] = act_full['pij2.weight']
                act_net['pij2.bias'] = act_full['pij2.bias']
                # act_net['vj2.weight'] = act_full['vj2.weight']
                # act_net['vj2.bias'] = act_full['vj2.bias']
                # act_net['pi2.weight'] = act_full['pi2.weight']
                # act_net['pi2.bias'] = act_full['pi2.bias']
                # act_net['v2.weight'] = act_full['v2.weight']
                # act_net['v2.bias'] = act_full['v2.bias']
            else:

                act_net = act_full
            # print(f2)

            # if IS_TEST:

            #    gnet = act_full
            # else:

            self.lnet.load_state_dict(act_net, strict=stric)
            # gnet.load(act_net, strict=stric)
            """
        self.my_q = my_queue
        self.my_ju = p_queue
        self.my_as = my_p2
        # self.my_tra = my_p3
        # self.game = SViz(use_mock, use_novel, level, True, seed, difficulty)
        self.game = vzd.DoomGame()
        self.step_limit = 1500  # 2100
        self.info_list = info_list
        # Sets path to additional resources wad file which is basically your scenario wad.
        # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
        # self.game.set_doom_scenario_path("../../scenarios/basic.wad")
        self.game.set_doom_scenario_path("../../scenarios/deadly_hall.wad")
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
        # if self.name == "w00":
        #    self.gaome.set_window_visible(True)
        # else:
        self.game.set_window_visible(DEBUG)  # IS_TEST)

        # Sets the living (for each move) to -1
        # self.game.set_living_reward(-1)
        self.game.set_death_penalty(50)
        self.total_target_count = 0
        self.max_target_count = 8

        # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
        self.game.set_mode(vzd.Mode.PLAYER)

        # Initialize the game. Further configuration won't take any effect from now on.
        self.game.init()

    def breaker(self, state, temp):
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
                # if IS_TEST and temp:
                #   print(o.name)
            elif o.name == "Clip":
                items.append(o)
            # else:
            #    print(o.name)
        # print(len(enemies))
        # exit()

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
        n_act = t[np.random.randint(0, 4)]  # 7#7  # navigate(armor, player)
        c_act = 7

        if clip:
            r_act = get_armor(clip, player)  # navigate(clip, player)

        if armor:
            m_act = navigate(armor, player)  # get_armor(armor, player)  # move(armor, player)
            # if not target and e_count > 0:
            #    n_act = m_act
            # n_act = get_armor(armor, player)  # navigate(armor, player)
            # n_act = hunt(armor, player)
        if target:
            # n_act = my_nact(player)#move(target, player)
            c_act = fight(target, player)  # gunner(target, player)

            # n_act = fight(target, player)
            # n_act = 7
            # print(n_act)
            # print(c_act)
            # exit()

        return np.asarray(sensor_vec), e_count, r_act, c_act, n_act, m_act, health, ammo, dist

    def run(self):  # bookmark

        total_step = 1
        actions = [[True, False, False, False, False, False, False], [False, True, False, False, False, False, False],
                   [False, False, True, False, False, False, False], [False, False, False, True, False, False, False],
                   [False, False, False, False, True, False, False], [False, False, False, False, False, True, False],
                   [False, False, False, False, False, False, True], [False, False, False, False, False, False, False]]
        actions2 = ['left', 'right', 'shoot', 'forward', 'backward', 'turn_left', 'turn_right', 'nothing']
        v_count = 0
        i3 = STATE_SIZE - 1  # 1 # 3
        i4 = STATE_SIZE - 3  # 3 # 2
        i5 = STATE_SIZE - 2  # 2 # 1
        i6 = STATE_SIZE - 4
        ep = 0
        task_var = 0.0
        r_list = []
        r_list2 = []
        t_kills = 0
        alist = ["c", "n", "r", "m"]

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
                state, True)  # initial state_vec
            # if IS_TEST:
            #    print(e_count)
            t_count = e_count

            buffer_s, buffer_a, buffer_r = [], [], []

            a_count = 0

            # task_var = 1.0
            # task_var = np.random.randint(0,4)
            state_vec[i3] = task_var  # 0.0#1.0

            if e_count > 0:  # can attack

                if c_act < 7:  # c_act < 7:
                    state_vec[i4] = 3.0
                else:
                    state_vec[i4] = 1.0

            state_vec[i5] = 0.0
            if r_act < 7:
                state_vec[i5] = 2.0
            state_vec[i6] = 1.0

            ep_r = 0.0
            ep_rr = 0.0
            ep += 1
            tk = 0
            fired = 0

            while True:
                step += 1
                reward = -1
                my_act2 = 'nothing'

                act = self.lnet.choose_action(v_wrap(state_vec[None, :]))

                # act = 2
                # if n_act < 7:
                #    act = 1
                # if c_act < 7:
                #    act = 0
                # elif r_act < 7:
                #    act = 1

                """
                if step < 15 and self.g_ep.value < 4:
                    if self.name == "w01" and step < 10:
                        #act = 2
                        if n_act < 7:
                            act = 0
                    elif self.name == "w00":
                        act = 2
                """
                """
                if self.name == "w00" and DEBUG:
                    print("act:", alist[act])
                    print("c:", actions2[c_act])
                    print("n:", actions2[n_act])
                    print("r:", actions2[r_act])
                    print("m:", actions2[m_act])
                    print("!!!!!!!")
                    sleep(0.05)
                """
                """
                if act == 0:
                    print("act", str(act), "c_act", str(c_act))
                else:
                    print(act)
                """
                # readjust = False
                # if step > 100 and t_kills > 200 and v_count < 20:
                #    readjust = True

                if act == 0:

                    # if c_act >= len(actions):
                    #    print(c_act)
                    #    print(len(actions))
                    my_act = actions[c_act]
                    my_act2 = actions2[c_act]
                    if c_act == 7:
                        reward -= 2
                    else:
                        reward += 1
                else:
                    if act == 1:
                        my_act = actions[n_act]
                        my_act2 = actions2[n_act]
                        if e_count == 0 or c_act < 7:
                            reward -= 1
                        # if reward
                        # if step > 1500 and state_vec[i4] == 3.0 and kills < 1:
                        #    reward -= 1

                        # print("nav")
                    elif act == 2:
                        # print("ammo")

                        if r_act < 7:
                            my_act = actions[r_act]
                            my_act2 = actions2[r_act]
                        else:

                            reward -= 2  # 3
                            # if step > 1000:
                            #    reward -= 1
                            my_act = actions[7]  # 7
                    else:
                        if m_act < 7:
                            my_act = actions[m_act]
                            my_act2 = actions2[m_act]
                        else:
                            reward -= 2
                            # if step > 1000:
                            #    reward -= 1
                            my_act = actions[7]  # 7

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                skiprate = 1
                # if my_act2 == "shoot":
                #    skiprate = 4

                rew = self.game.make_action(my_act, skiprate)
                victory = False
                new_state = self.game.get_state()
                done = self.game.is_episode_finished()
                dead = self.game.is_player_dead()

                if rew > 1:
                    # print(step)
                    victory = True
                elif done and not dead:
                    reward -= 5

                reward += rew

                # print(reward)

                # if new_state.objects:
                nstate_vec = []
                if not done:
                    pact = c_act
                    nstate_vec, e_temp, r_act, c_act, n_act, m_act, h, n_ammo, dist = self.breaker(new_state, False)

                    # print(health)
                    # exit()

                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    if dist < o_dist and act > 0:
                        # if act == 3:
                        # if act == 3 and c_act == 7:
                        #    #if not act == 1 or c_act == 7:
                        reward += 0.5  # 0.2#(0.1 + (0.1 * kills))#0.5

                        """
                            #if kills > 1 or dist > 400:
                            #    #if c_act < 7:# and not readjust:
                            #    #    reward += 0.25
                            #    #else:
                            #    reward += 0.5


                            #if act == 3:
                            #    if pact < 7:
                            #        reward += 0.5
                            #    else:
                            #        reward += 0.25
                        else:
                            reward += 0.1
                        """

                    o_dist = dist
                    if n_ammo < ammo:
                        fired += 1
                    if e_temp < e_count:
                        # e_count -= 1
                        if act == 0 and pact < 7:
                            # print("hi")
                            # exit()
                            # if not readjust:
                            reward += 15  # 25#10
                            # else:
                            #    reward += 50
                            kills += 1
                            tk += 1
                            # fired += 1
                            # print(fired)

                            fired = 0
                        else:
                            tk += 1
                            # print("rip:", str(n_ammo))
                            # reward += 10
                    e_count = e_temp

                    if h < health:
                        # if (act != 0 or c_act == 7):
                        reward -= 1.0  # 0.5
                        # else:
                        #    reward -= 0.5

                    health = h

                    if n_ammo > ammo:
                        reward += 10

                    ammo = n_ammo

                    nstate_vec[i3] = task_var  # 0.0#1.0
                    nstate_vec[i4] = 0.0
                    if e_count > 0:
                        if c_act < 7:

                            nstate_vec[i4] = 3.0

                        else:
                            nstate_vec[i4] = 1.0

                    nstate_vec[i5] = 0.0
                    if r_act < 7:
                        nstate_vec[i5] = 2.0
                    nstate_vec[i6] = 1.0
                    if victory:
                        nstate_vec[i6] = 0.0
                    elif dist < 200:
                        nstate_vec[i6] = 2.0


                else:
                    nstate_vec = state_vec
                    if dead:
                        nstate_vec[4] = 0
                arm = 2
                if victory:
                    v_count += 1
                    # reward += (kills * 20)
                    done = True

                    arm = 0
                current_targets = 0
                current_targets = current_targets + (t_count - kills) + arm  # e_count
                self.total_target_count = self.total_target_count + current_targets
                target_by_time = current_targets * (self.step_limit - step)
                performance = 1 - (self.total_target_count + target_by_time) / (
                        self.step_limit * self.max_target_count)
                performance = round(performance, 6)

                ep_r = performance  # step  # performance  # reward
                ep_rr += reward

                # if done and step < 2099 and not dead:
                #    v_count += 1
                #    reward += 20

                if not IS_TEST:
                    buffer_a.append(act)
                    buffer_s.append(state_vec)
                    buffer_r.append(reward)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync

                    if len(buffer_s) > 0 and not IS_TEST:
                        push_and_pull(self.opt, self.lnet, self.gnet, done, nstate_vec, buffer_s, buffer_a, buffer_r,
                                      GAMMA)

                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        t_kills += kills
                        if IS_TEST:
                            r_list.append(ep_r)
                            r_list2.append(ep_rr)
                            self.info_list.put([self.g_ep.value, ep_rr, step, performance, kills, a_count])
                        else:
                            # self.my_as.put(ep_rr)
                            # self.my_tra.put(ep_r)
                            self.info_list.put([self.g_ep.value, ep_rr, step, performance, kills, a_count])
                        record_dead(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name, t_count, kills, victory,
                                    dead, a_count, self.my_ju, o_dist, step, tk, self.my_as)
                        break

                state_vec = nstate_vec
                state = new_state
                total_step += 1
            # turn += 1
            # self.game.close()

        if IS_TEST:
            print(v_count)
            # print(np.average(r_list))
            av = np.average(r_list)
            print(av)
            self.my_q.put([v_count, np.average(r_list2), av])
        self.my_ju.put(None)
        # self.my_tra.put(None)
        self.my_as.put(None)
        self.info_list.put(None)
        self.res_queue.put(None)


def main(f, my_q, my_r, f2, my_q2, f3):
    gnet = Net(STATE_SIZE, ACTION_SIZE)  # global network    global_kills = mp.Value('i', 0)
    # gnet.share_memory()  # share the global parameters in multiprocessing
    # start = timeit.timeit()

    # end = timeit.timeit()
    # print(end - start)
    # exit()
    # l = "Y"  # input("load Y or N:")

    my_jump = mp.Queue()  # mp.Queue()
    my_asym = mp.Queue()
    my_info = mp.Queue()
    # my_tran = mp.Queue()
    l = "Y"
    stric = True
    if IS_TEST:
        l = "Y"
        stric = True
    act_net = {}
    print(f)
    # if l == "Y":
    # act_net = torch.load(f)
    # gnet.load_state_dict(act_net, strict=stric)
    if l == "Y":

        act_full = torch.load(f)

        if not stric:  # critic layers 1-3 work kind of

            act_net['pi1.weight'] = act_full['pi1.weight']
            act_net['pi1.bias'] = act_full['pi1.bias']
            act_net['v1.weight'] = act_full['v1.weight']
            act_net['v1.bias'] = act_full['v1.bias']
            act_net['pij.weight'] = act_full['pij.weight']
            act_net['pij.bias'] = act_full['pij.bias']
            act_net['vj.weight'] = act_full['vj.weight']
            act_net['vj.bias'] = act_full['vj.bias']
            act_net['pij2.weight'] = act_full['pij2.weight']
            act_net['pij2.bias'] = act_full['pij2.bias']
            act_net['vj2.weight'] = act_full['vj2.weight']
            act_net['vj2.bias'] = act_full['vj2.bias']
            # act_net['pi2.weight'] = act_full['pi2.weight']
            # act_net['pi2.bias'] = act_full['pi2.bias']
            # act_net['v2.weight'] = act_full['v2.weight']
            # act_net['v2.bias'] = act_full['v2.bias']
        else:

            act_net = act_full

        # print(f2)

        # if IS_TEST:

        #    gnet = act_full
        # else:
        # act_net = act_full
        gnet.load_state_dict(act_net, strict=stric)
        # gnet.load(act_net, strict=stric)

    my_lr = 1e-3
    if IS_TEST:
        my_lr = 1e-4
    opt = SharedAdam(gnet.parameters(), lr=my_lr, betas=(0.90, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    cp_count = 1
    if IS_TEST:
        print("testing")
    else:
        print("training")
        cp_count = 4
        if DEBUG:
            cp_count = 1

    # parallel training
    if mp.cpu_count() < 6:
        print("cpu alert")
        exit()

    workers = [
        Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, act_net, stric, my_q, my_jump, my_asym, my_info, l) for
        i in
        range(cp_count)]

    [w.start() for w in workers]
    res = []  # record episode reward to plot

    m_jump = 0
    # m_tran = 0
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
        transrate = []

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
        # while True:
        #    p = my_tran.get()
        #    if p is not None:
        # transrate.append(p)
        #    else:
        #        break
        # m_tran = np.average(transrate)
    myinfo = []  # f3 = f0 + "raw.csv"
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
    [w.join() for w in workers]

    if not IS_TEST:
        print(f2)
        torch.save(gnet.state_dict(), f2)
        # torch.save(gnet, f2)

        my_r2 = np.add(my_r, res)
        my_q3 = my_q2
        my_q3.append([m_jump, m_asym])

        return my_r2, my_q3
    return my_r, my_q2


if __name__ == "__main__":
    x = 0
    rang = 30
    test_ep = 1000
    # MAX_EP = 10 #5

    control = "Y"  # "Y"#input("control Y or N:")
    # testing = "N"  # input("test Y or N:")
    is_load = "N"  # input("continue Y or N:")

    if control == "Y" or control == "y":
        IS_CONTROL = True
    # if testing == "Y" or testing == "y":
    #    IS_TEST = True

    # if IS_TEST:
    #    MAX_EP = test_ep

    my_q = mp.Queue()
    my_q2 = []
    my_r = np.zeros([MAX_EP])
    pref = "boot_"
    if IS_CONTROL:
        pref = "control_"

    for ind in range(rang):
        n = ind + x
        f0 = pref + str(n)
        f1 = f0 + ".txt"
        f2 = pref + "deadly2_" + str(n) + ".txt"
        # print(f2)
        f3 = f0 + "raw2.csv"
        if not os.path.exists(f1):
            print("file:", f1, "does not exist")
            break
        # print(f1)
        my_r, my_q2 = main(f1, my_q, my_r, f2, my_q2, f3)  # my_jump, my_asym, my_tran)

    IS_TEST = True
    # DEBUG = True
    # if IS_TEST:
    MAX_EP = test_ep

    my_q = mp.Queue()
    f2 = "dud.txt"

    for ind in range(rang):
        n = ind + x
        f0 = pref + str(n)
        f1 = pref + "deadly2_" + str(n) + ".txt"
        f3 = f0 + "rawtest2.csv"
        if not os.path.exists(f1):
            print("file:", f1, "does not exist")
            break
        # print(f1)
        _, _ = main(f1, my_q, my_r, f2, my_q2, f3)  # my_jump, my_asym, my_tran)
    # name of csv file
    filename = "boot_deadly2.csv"

    if IS_CONTROL:
        filename = "control_deadly2.csv"

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

    f = open("myout_deadly2.txt", "w")
    # if IS_TEST:
    if control == "N":
        f.write("boot\n")
    else:
        f.write("control\n")
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

    """
    # if IS_TEST:
    while True:
        r = my_q.get()
        if r is not None:
            print(r)

        else:
            break

    print("other")
    for i in my_q2:
        print(i)
    """
"""
[21, -245.26, 0.31267835]
[57, -94.72, 0.40947502999999996]
[8, -374.28, 0.34252250000000006]
[5, -2501.81, 0.42222746]
[39, -1035.195, 0.33509919000000005]
[0, -1370.76, 0.2425]
[22, -137.43, 0.40386669]
[2, -3496.955, 0.3964366]
[0, -1555.18, 0.28050167000000004]
[0, -1939.065, 0.28248249]
other
[0.36889361306532664, -87.7835, 0.36630200299999993]
[0.3702344974874372, -84.8815, 0.37038749199999993]
[0.405249175879397, -146.1075, 0.38100501400000003]
[0.34887939195979895, -103.1475, 0.37027192099999995]
[0.3685837537688442, -83.761, 0.387979498]
[0.26129982914572863, -468.448, 0.33524466599999997]
[0.6312399045226131, -212.4875, 0.568686604]
[0.5085782462311558, -254.4235, 0.5882093079999999]
[0.5976222261306533, -267.014, 0.5757090559999999]
[0.3866025527638191, -583.574, 0.380750408]

"""