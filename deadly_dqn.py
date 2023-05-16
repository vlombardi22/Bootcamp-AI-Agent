# https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_cartpole.py
# this is the good one
import numpy as np

import random
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from boot_utils.hall_skills import get_dist, get_angle
import vizdoom as vzd

TRAIN_EPS = 1000
TEST_EPS = 1000
IS_TEST = False


# Wrapper class for keras-rl dqn learning
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




class CWrapper:

    def __init__(self, my_res, my_jump, my_asym, my_info, seed=97):
        # Parameters

        self.seed = seed
        self.my_res = my_res
        self.metrics = [0.0, 0.0, 0.0]
        self.kills = 0
        self.step_limit = 1500
        # Internal vars
        self.state = None
        self.my_pref = 0
        self.my_jump = my_jump
        self.my_asym = my_asym
        self.my_info = my_info
        self.ep_count = 0
        self.ep_reward = 0
        self.step_count = 0
        self.last_health = None
        self.walls = None
        self.a_count = 0
        self.h_count = 0
        self.e_count = 0
        self.dist = 0
        self.obs = None

        self.total_target_count = 0
        self.max_target_count = 8
        self.seed_list = []
        # REQUIRED:
        self.game = vzd.DoomGame()
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
        self.action_space = range(8)
        self.observation_space = np.zeros(25)


    def wipe(self, my_info, testing):
        self.my_info = my_info
        if testing:
            self.metrics = [0.0, 0.0, 0.0]
    def analyze(self):
        self.metrics[1] = float(self.metrics[1] / TEST_EPS)
        self.metrics[2] = float(self.metrics[2] / TEST_EPS)
        return self.metrics

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
        #c_act = 0 # acts as a sight byt did not help here
        #if target:
        #    c_act = 1
        sensor_vec = [player.position_x, player.position_y, player.angle, float(ammo), float(health),
                      e_count] + strat_enemy + [
                         i_count] + strat_item + [a_count] + strat_armor + [0.0, 0.0, 0.0, 0.0]

        return np.asarray(sensor_vec), e_count, a_count, i_count, health, ammo, dist

    def set_seed(self):
        self.ep_count = 0
        seed = 97
        random.seed(seed)
        np.random.seed(seed)
        self.seed_list = [np.random.randint(0, 1000) for i in range(TEST_EPS)]

    def step(self, action):
        actions = [[True, False, False, False, False, False, False], [False, True, False, False, False, False, False],
                   [False, False, True, False, False, False, False], [False, False, False, True, False, False, False],
                   [False, False, False, False, True, False, False], [False, False, False, False, False, True, False],
                   [False, False, False, False, False, False, True], [False, False, False, False, False, False, False]]
        rew = self.game.make_action(actions[action], 1)
        victory = False

        state = self.game.get_state()
        if not state:
            state = self.state
        self.state = state

        obs, e_count, a_count, i_count, _, _, dist = self.breaker(state)

        done = self.game.is_episode_finished()
        dead = self.game.is_player_dead()
        reward = -1
        if rew > 1:
            victory = True
        elif done and not dead:
            reward -= 5

        task = 0

        health = 0
        health = health + e_count

        h_count = i_count

        if dead:
            reward = -300
        elif victory:
            reward = 500
        else:
            reward = -1

        if self.last_health is not None:
            if health < self.last_health:
                reward += 25
        if dist < self.dist:
            reward += 1
        if e_count < self.e_count:
            reward += 200
            self.kills += 1

        if h_count < self.h_count:
            reward += 500

        if a_count < self.a_count:
            reward += 200

        arm = 2
        if victory:

            arm = 0
        current_targets = 0
        current_targets = current_targets + (6 - self.kills) + arm  # e_count
        self.total_target_count = self.total_target_count + current_targets
        target_by_time = current_targets * (self.step_limit - self.step_count)
        performance = 1 - (self.total_target_count + target_by_time) / (
                self.step_limit * self.max_target_count)
        pref = round(performance, 6)

        self.ep_reward += reward
        if done:
            if not IS_TEST:

                p = pref
                if self.step_count > 0:
                    p = self.my_pref * 0.99 + pref * 0.01
                self.my_pref = p

                if self.ep_count >= len(self.my_res):
                    self.my_res.append(p)
                else:
                    self.my_res[self.ep_count] += p

                if self.ep_count < 200:
                    self.my_jump.append(pref)
                if self.ep_count >= 800:
                    self.my_asym.append(pref)

            else:
                if victory:
                    self.metrics[0] += 1
                self.metrics[1] += reward
                self.metrics[2] += pref

            self.my_info.append([self.ep_count, self.ep_reward, self.step_count, pref, self.kills, a_count, h_count])
        self.e_count = e_count
        self.a_count = a_count
        self.h_count = h_count
        self.dist = dist
        self.last_health = health
        self.step_count += 1

        return obs, reward, done, {}

    def reset(self):
        self.last_health = None
        self.step_count = 0

        self.ep_reward = 0

        if IS_TEST:
            seed = self.seed_list[self.ep_count]
            self.game.set_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.game.new_episode()
        self.ep_count += 1

        ob, e_count, a_count, i_count, _, _, dist = self.breaker(self.game.get_state())

        self.kills = 0
        self.h_count = i_count
        self.a_count = a_count
        self.e_count = e_count
        self.dist = dist
        self.total_target_count = 0
        return ob


if __name__ == "__main__":
    train = True
    IS_TEST = False
    my_res = []

    jump = []
    asympt = []
    my_info = []
    train_metrics = []
    my_results = []

    rang = 30
    is_load = "N"
    env = CWrapper(my_res, jump, asympt, my_info, seed=97)
    nb_actions = len(env.action_space) - 1
    step_length = 1500

    targ_steps = step_length * TRAIN_EPS

    for i in range(rang):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics! 50000
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                       target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

        # Ctrl + C. 150000
        # 3000000
        my_str = "vizdoom_task123" + str(i) + ".h5f"

        dqn.fit(env, nb_steps=targ_steps, visualize=False, verbose=2, nb_max_episode_steps=step_length)
        my_str2 = "vizdoom_task6" + str(i) + ".h5f"
        f3 = "vizdoom_task6" + str(i) + "raw.csv"
        with open(f3, 'w', newline='') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(my_info)
        csvfile.close()

        # After training is done, we save the final weights.
        dqn.save_weights(my_str2, overwrite=True)
        m_asym = np.average(asympt)
        m_jump = np.average(jump)
        train_metrics.append([m_jump, m_asym])
        env.ep_count = 0
        my_info = []
        env.wipe(my_info, False)
        # Finally, evaluate our algorithm for 5 episodes.

    IS_TEST = True
    for i in range(rang):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        # print(model.summary())

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics! 50000
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        env.set_seed()
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                       target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
        my_str = "vizdoom_task6" + str(i) + ".h5f"
        f3 = "vizdoom_task6" + str(i) + "rawtest.csv"
        dqn.load_weights(my_str)

        t = dqn.test(env, nb_episodes=TEST_EPS, visualize=False)
        with open(f3, 'w', newline='') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(my_info)
        csvfile.close()
        my_results.append(env.analyze())
        my_info = []
        env.wipe(my_info, True)

    filename = "dqn_task6.csv"

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
    print(len(my_res))
    f = open("myout_dqn_task6.txt", "w")
    f.write("dqn\n")
    for r in my_results:
        mystr = str(r) + "\n"
        f.write(mystr)

    f.write("other\n")
    print("other")
    for i in train_metrics:
        mystr2 = str(i) + "\n"
        f.write(mystr2)
        print(i)
    f.close()
    print("done")