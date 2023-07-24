import os
# import copy
import time
import random

import numpy as np
import vizdoom as vzd

from boot_utils.Agents import Agents


class SailonViz:

    def __init__(self, use_mock, use_novel, level, use_img, seed, difficulty,
                 path="env_generator/envs/", use_gui=False, use_seed=False, task=1, base=False):

        # Attempt to clear out previous ini file
        try:
            os.remove("_vizdoom.ini")
        except FileNotFoundError as e:
            None

        # Set external to internal parameters
        self.use_mock = use_mock
        self.use_novel = use_novel
        self.level = level
        self.use_img = use_img
        self.seed = seed
        self.difficulty = difficulty
        self.path = path
        self.use_seed = use_seed

        self.total_target_count = 0
        self.max_target_count = 2
        if task > 3:
            self.max_target_count = 3
        # Declare parameters
        self.tick = None
        self.performance = None
        self.last_obs = None
        self.enemies_health = None
        self.viz_id_to_cvar = dict()
        self.use_top_down = True
        self.walls = None
        self.time = None
        self.time_delta = 1.0 / 35.0
        self.last_health = None
        self.task = task
        # REAL NOVELTY VALS REMOVE IN PUBLIC
        self.enemy_num = 4
        self.lives = 4

        self.use_new_spawn = 1
        # REAL NOVELTY IS REVEALED HERE REMOVE BEFORE SHIPPING
        if self.level == 208:
            self.max_enemy_health = 8 * 10
        else:
            self.max_enemy_health = 4 * 10
        self.total_enemy_health = None

        # Set internal params

        self.step_limit = 20#1500
        if self.task > 3:
            self.step_limit = 20 #2000
        self.actions = {'nothing': [False, False, False, False, False, 0],
                        'left': [True, False, False, False, False, 0],
                        'right': [False, True, False, False, False, 0],
                        'backward': [False, False, True, False, False, 0],
                        'forward': [False, False, False, True, False, 0],
                        'shoot': [False, False, False, False, True, 0],
                        'turn_left': [False, False, False, False, False, -45],
                        'turn_right': [False, False, False, False, False, 45]}

        if self.level == 53:
            if self.difficulty == 'easy':
                buckets = 50
            if self.difficulty == 'medium':
                buckets = 40
            if self.difficulty == 'hard':
                buckets = 30

            self.bucket_vals = np.arange(-512, 512, step=1023.9 / buckets)

        # Convert string difficulty to number
        conv = {'easy': 1, 'medium': 2, 'hard': 3}
        difficulty_str = str(conv[self.difficulty])

        # Decide on agent behaviour here
        temp = False
        if task > 3 and base:
            temp = True
        self.Agents = Agents(self.level, self.difficulty, self.use_mock, temp)

        # Make and load game parameters here

        game = vzd.DoomGame()
        package_path = ''  # self.path + "vizdoom/"
        game.load_config(package_path + 'scenarios/basic.cfg')
        scenario = ""
        if task <= 1:
            if base:
                scenario = "scenarios/phase_4_cwall.wad"
            else:
                scenario = "scenarios/phase_4_combat.wad"

        if task == 2:
            if base:
                scenario = "scenarios/phase_4_awall.wad"
            else:
                scenario = "scenarios/phase_4_ammo.wad"
        elif task == 3:
            if base:
                scenario = "scenarios/phase_4_hwall.wad"
            else:
                scenario = "scenarios/phase_4_health.wad"
        elif task > 3:

            if self.level == 208:
                if base:
                    scenario = "scenarios/phase_4_twall5.wad"
                else:
                    scenario = "scenarios/phase_4_test5.wad"
            else:
                if base:
                    scenario = "scenarios/phase_4_twall4.wad"
                else:
                    scenario = "scenarios/phase_4_test4.wad"
        game.set_doom_scenario_path(package_path + scenario)

        # Set in game limit
        game.set_episode_timeout(self.step_limit)

        # Set use gui
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        if use_gui:
            # game.set_screen_resolution(vzd.vizdoom.RES_640X400)
            game.set_window_visible(use_gui)

        # Set game tie ins
        game.add_available_game_variable(vzd.vizdoom.HEALTH)
        game.add_available_game_variable(vzd.vizdoom.AMMO2)

        # These are enemies health
        game.add_available_game_variable(vzd.vizdoom.USER20)
        game.add_available_game_variable(vzd.vizdoom.USER21)
        game.add_available_game_variable(vzd.vizdoom.USER22)
        game.add_available_game_variable(vzd.vizdoom.USER23)

        # Enemies x
        game.add_available_game_variable(vzd.vizdoom.USER30)
        game.add_available_game_variable(vzd.vizdoom.USER31)
        game.add_available_game_variable(vzd.vizdoom.USER32)
        game.add_available_game_variable(vzd.vizdoom.USER33)

        # Enemies y
        game.add_available_game_variable(vzd.vizdoom.USER40)
        game.add_available_game_variable(vzd.vizdoom.USER41)
        game.add_available_game_variable(vzd.vizdoom.USER42)
        game.add_available_game_variable(vzd.vizdoom.USER43)

        # Send level specific info here (filtered in wad to select right novelty)
        game.add_game_args("+set novelty " + str(self.level))
        game.add_game_args("+set difficulty_n " + str(difficulty_str))
        game.add_game_args("+set use_new_spawn" + str(self.use_new_spawn))

        # Enables information about all objects present in the current episode/level.
        game.set_objects_info_enabled(True)

        # Enables information about all sectors (map layout).
        game.set_sectors_info_enabled(True)

        # Set seed here
        if self.use_seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
            game.set_seed(self.seed)

        # Call this at the end since no more changes can be made after
        game.init()
        self.game = game

    def step(self, action):
        # Decode action
        action = self.actions[action]

        # Set agent behaviour before making the action (which calls an ingame update)
        # Returns a string array, use as commands in vizdoom
        comands = self.Agents.act(self.get_state())
        for command in comands:
            self.game.send_game_command(command)

        # Make action
        self.game.make_action(action)

        # Update counter
        self.tick = self.tick + 1

        # Get game state information
        observation = self.get_state()
        current_targets = 0
        if self.task <= 1 or self.task > 3:
            current_targets = current_targets + len(observation['enemies'])
        elif self.task == 2:
            current_targets = current_targets + len(observation['items']['ammo'])
        elif self.task == 3:
            current_targets = current_targets + len(observation['items']['health'])
        self.total_target_count = self.total_target_count + current_targets
        target_by_time = current_targets * (self.step_limit - self.tick)
        self.performance = 1 - (self.total_target_count + target_by_time) / (self.step_limit * self.max_target_count)
        self.performance = round(self.performance, 6)

        victory = False
        dead = False
        done = self.is_done()
        if self.game.is_player_dead():
            done = True
            dead = True

        # Special check to see if all monsters died (needed to prevent additional tick)
        if len(observation['enemies']) == 0 and (self.task <= 1 or self.task > 3):
            done = True
            victory = True

        # Update last obs
        self.last_obs = observation

        return observation, self.performance, done, victory, dead  # {}

    def is_done(self):
        # Get game state information
        observation = self.get_state()

        # Check if game is done naturally (includes tick limit natively)
        done = self.game.is_episode_finished()

        # Check if player died
        if self.game.is_player_dead():
            done = True

        # Special exit flag from viz script
        if self.Agents.special_exit_flag:
            done = True

        # Special check to see if all monsters died (needed to prevent additional tick)

        if len(observation['enemies']) == 0 and (self.task <= 1 or self.task > 3):
            done = True

        return done

    def get_time(self):
        return self.time + self.tick * self.time_delta

    def get_actions(self):
        return ['nothing', 'left', 'right', 'backward', 'forward', 'shoot', 'turn_left', 'turn_right']

    def get_image(self, episode):
        # Check to generate image or not
        if not self.use_img:
            return None

        if self.is_done():
            return None

        if self.game is None:
            return None

        # Top down
        if self.use_top_down:
            return self.get_top_down(episode)
        else:
            # Get current game information
            return self.game.get_state().screen_buffer

    def get_top_down(self, episode):
        import matplotlib.pyplot as plt
        # Make a random plot...
        fig2 = plt.figure()

        fig2.add_subplot(111)

        plt.xlim(-522, 522 + 300)
        plt.ylim(-522, 522)
        plt.gca().set_aspect('equal', adjustable='box')

        # Draw stuff
        state = self.get_state()
        for wall in self.walls:
            plt.plot([wall['x1'], wall['x2']], [wall['y1'], wall['y2']], color='black', linewidth=2)

        # Draw player
        x, y = state['player']['x_position'], state['player']['y_position']
        a = state['player']['angle'] / 180 * np.pi
        plt.plot(x, y, marker="$P$", label='player')
        plt.arrow(x, y, 75 * np.cos(a), 75 * np.sin(a), head_width=10)

        # Draw enemy
        for ind, enemy in enumerate(state['enemies']):
            x, y = enemy['x_position'], enemy['y_position']
            a = enemy['angle'] / 180 * np.pi
            if ind == 0:
                plt.plot(x, y, marker="$" + str(enemy['id']) + "$", color='red', label='enemy')
            else:
                plt.plot(x, y, marker="$" + str(enemy['id']) + "$", color='red')
            plt.arrow(x, y, 75 * np.cos(a), 75 * np.sin(a), head_width=10)

        # Draw stuff
        mark = {'health': "$H$", 'trap': "x", 'obstacle': 's', 'ammo': "$A$"}
        col = {'health': 'green', 'ammo': 'green', 'obstacle': 'black', 'trap': 'red'}
        for type in state['items']:
            for ind, item in enumerate(state['items'][type]):
                x, y = item['x_position'], item['y_position']
                if ind == 0:
                    plt.plot(x, y, marker=mark[type], color=col[type], label=type)
                else:
                    plt.plot(x, y, marker=mark[type], color=col[type])

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        plt.legend(loc='center right', handlelength=0)

        plt.ylabel('Y-axis')
        plt.xlabel('X-axis')

        fig2.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.fromstring(fig2.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig2.canvas.get_width_height()[::-1] + (3,))

        plt.savefig("ep_" + str(episode) + ".png")
        plt.close()

        return data

    def get_state(self, initial=False):
        # Check for game end, if so just send last value
        if self.game.is_episode_finished():
            return self.last_obs

        if self.game is None:
            return self.last_obs

        # Get current game information
        state = self.game.get_state()
        health = self.game.get_game_variable(vzd.vizdoom.HEALTH)
        ammo = self.game.get_game_variable(vzd.vizdoom.AMMO2)

        # This big block links the x,y of the enemies to id for health getting
        # Its convoluted but there were no other tie-ins :(
        if initial or self.level == 208:
            self.enemies_health = dict()
            game_vars = [vzd.vizdoom.USER20, vzd.vizdoom.USER21,
                         vzd.vizdoom.USER22, vzd.vizdoom.USER23]

            x_vars = [vzd.vizdoom.USER30, vzd.vizdoom.USER31,
                      vzd.vizdoom.USER32, vzd.vizdoom.USER33]

            y_vars = [vzd.vizdoom.USER40, vzd.vizdoom.USER41,
                      vzd.vizdoom.USER42, vzd.vizdoom.USER43]

            for object in state.objects:
                if "Zombie" in object.name and "Dead" not in object.name:
                    for i in range(4):
                        x_pos = self.game.get_game_variable(x_vars[i])
                        y_pos = self.game.get_game_variable(y_vars[i])
                        dif = abs(x_pos - object.position_x) + abs(y_pos - object.position_y)
                        if dif < 5:
                            self.enemies_health[object.id] = game_vars[i]
                            self.viz_id_to_cvar[object.id] = i + 1

        # Start formatting the data
        data = {'enemies': [], 'items': {'health': [], 'ammo': [], 'trap': [], 'obstacle': []}}
        for object in state.objects:
            # Base entity information
            if self.level != 53:
                entity = {'id': int(object.id),
                          'angle': round(object.angle, 4),
                          'x_position': round(float(object.position_x), 4),
                          'y_position': round(float(object.position_y), 4),
                          'z_position': round(float(object.position_z), 4)}
            # This is for bucketing phase 1 level 3
            else:
                entity = {'id': int(object.id),
                          'angle': round(object.angle, 4),
                          'x_position': round(float(object.position_x), 4),
                          'y_position': round(float(object.position_y), 4),
                          'z_position': round(float(object.position_z), 4)}

                entity['x_position'] = self.bucket_vals[(np.abs(self.bucket_vals - entity['x_position'])).argmin()]
                entity['x_position'] = round(entity['x_position'], 4)
                entity['y_position'] = self.bucket_vals[(np.abs(self.bucket_vals - entity['y_position'])).argmin()]
                entity['y_position'] = round(entity['y_position'], 4)

            # Check for player
            if object.name == 'Doomer':
                entity['health'] = float(health)
                entity['ammo'] = float(ammo)
                data['player'] = entity

            # Check for enemy
            elif "Zombie" in object.name and "Dead" not in object.name and object.id in self.enemies_health.keys():
                entity['name'] = "ZombieMan"
                entity['health'] = self.game.get_game_variable(self.enemies_health[int(object.id)])
                data['enemies'].append(entity)

            # Whiskey is the ultimate trap!
            elif "Whiskeyy" in object.name:
                data['items']['trap'].append(entity)

            elif "Health" in object.name:
                data['items']['health'].append(entity)

            elif "Clip" in object.name:
                data['items']['ammo'].append(entity)

            elif "TallRedColumn2" in object.name:
                data['items']['obstacle'].append(entity)

        # Get lines
        if initial:
            data['walls'] = []
            for sector in state.sectors:
                if sector.ceiling_height < 10.0:
                    continue
                for line in sector.lines:
                    data['walls'].append({'x1': round(line.x1, 2), 'x2': round(line.x2, 2),
                                          'y1': round(line.y1, 2), 'y2': round(line.y2, 2)})
            self.walls = data['walls']

        if initial:
            self.Agents.walls = self.walls
            self.Agents.viz_id_to_cvar = self.viz_id_to_cvar

        return data

    def reset(self, seed=0):
        # Reset params
        self.tick = 0
        self.performance = None
        self.last_obs = None
        self.total_enemy_health = 0
        self.time = time.time()
        self.total_target_count = 0
        # Start a new episode
        self.game.new_episode()

        # Docs suggest putting it here too
        # self.game.set_seed(self.seed)
        if self.use_seed:
            self.game.set_seed(seed)

        # Get state
        observation = self.get_state(initial=True)

        return observation
