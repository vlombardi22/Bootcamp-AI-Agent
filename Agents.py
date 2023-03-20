import random

import numpy as np

FORWARD = 1
BACKWARD = 2
LEFT = 3
RIGHT = 4
TLEFT = 5
TRIGHT = 6
SHOOT = 7
NOTHING = 8


class Agents:

    def __init__(self, level, difficulty, mock):
        # level and novelty selections
        self.level = level
        self.difficulty = difficulty
        self.mock = mock

        # Set looking bounds (vision cone)
        self.left_side = np.pi * 7 / 8
        self.right_side = np.pi / 8

        self.viz_id_to_cvar = None

        # Used for checking wall collision
        self.walls = None

        # Used for check
        self.last_dist = np.zeros((4, 4))

        # Abandoned novelties
        # For un-used novelty
        self.hunger_games_tick = None

        # Mock novelties
        # Used for revealed novelty 7
        self.covers = None

        # REAL NOVELTIES BELOW
        # Used for real novelty 3
        self.last = [10] * 4
        self.lastlast = [10] * 4

        # Used for real novelty 5
        self.hunting = False
        self.tick_counter = -1
        self.hunt_tick = None

        # Used for real novelty 7
        self.special_exit_flag = False

        return

    # Run agent behavoir here
    def act(self, state):
        # Used for certain novelties
        self.tick_counter = self.tick_counter + 1

        # Sort lists so its always
        state['enemies'] = sorted(state['enemies'], key=lambda d: d['id'])
        state['items']['obstacle'] = sorted(state['items']['obstacle'], key=lambda d: d['id'])

        # Choose what to do based off novelty level
        commands = list()

        # Enemies move towards player
        if self.level == 103:
            commands = self.move_towards(state)

        # Enemies move away from avg
        elif self.level == 105:
            commands = self.spread_out(state)

        # Enemies move away from avg
        elif self.level == 107:
            commands = self.take_cover(state)

        # Enemies move away from player
        elif self.level == 203:
            commands = self.teleport(state)

        # Enemies switch between moving and shooting
        elif self.level == 205:
            commands = self.hunt(state)

        # Enemies move to point to win
        elif self.level == 207:
            commands = self.point_defense(state)

        # Any other is pure random base
        else:
            for ind, val in enumerate(state['enemies']):
                action = random.choice([FORWARD, BACKWARD, LEFT, RIGHT, TLEFT, TRIGHT, SHOOT])
                commands.append([ind, action])

        # Enemies always check for facing to see if shoot
        if self.level != 207:
            commands = self.check_shoot(state, commands)

        # Enemies never shoot towards other enemies
        commands = self.check_enemies(state, commands)

        # Formats the string for vizdoom
        cvar_cmd_str = self.format_string(state, commands)

        return cvar_cmd_str

    # Add vizdoom specific info to commands
    def format_string(self, state, commands):
        str_commands = []
        for command in reversed(commands):
            # fill in string with correct ai num and action
            # looks like --> 'set ai_3 7'
            str_command = 'set ai_{} {}'.format(self.viz_id_to_cvar[state['enemies'][command[0]]['id']], command[1])
            str_commands.append(str_command)
        return str_commands

    # Novelty 103
    # Move agent towards player
    def move_towards(self, state):
        commands = []

        for ind, val in enumerate(state['enemies']):
            # Get info
            angle, sign = self.get_angle(state['player'], val)

            if angle < self.right_side:
                action = random.choice([FORWARD, LEFT, RIGHT, SHOOT])
            else:
                if sign == -1.0:
                    action = TRIGHT
                else:
                    action = TLEFT

            # Send ai action
            commands.append([ind, action])

        return commands

    # Novelty 105
    def spread_out(self, state):
        # Response commands
        commands = []

        # Find avg pos
        avg_x = 0.0
        avg_y = 0.0
        for ind, val in enumerate(state['enemies']):
            avg_x = avg_x + val['x_position']
            avg_y = avg_y + val['y_position']

        avg_x = avg_x / len(state['enemies'])
        avg_y = avg_y / len(state['enemies'])

        # Do spread out logic
        for ind, val in enumerate(state['enemies']):
            # Get info
            pl = {'x_position': avg_x, 'y_position': avg_y}

            angle, sign = self.get_angle(pl, val)

            # If enemy is face towards player turn away
            if angle > self.left_side:
                # Forward, left, right, shoot?
                action = random.choice([FORWARD, LEFT, RIGHT, SHOOT])
            else:
                if sign == 1.0:
                    # Turn right
                    action = TLEFT
                else:
                    # Turn left
                    action = TRIGHT

            # Send ai action
            commands.append([ind, action])

        return commands

    # Novelty 107
    # Enemies move away from player behind cover
    def take_cover(self, state):
        commands = []
        cover_dist = 50

        # Assign closet obstacle to agent
        if self.covers is None:
            self.covers = {}
            for en_ind, enemy in enumerate(state['enemies']):
                enemy_pos = np.asarray([enemy['x_position'], enemy['y_position']])
                min_dist = None

                for obs_ind, obstacle in enumerate(state['items']['obstacle']):
                    obs_pos = np.asarray([obstacle['x_position'], obstacle['y_position']])
                    dist = np.linalg.norm(enemy_pos - obs_pos)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        self.covers[en_ind] = obs_ind

        # TODO: This is default goto script, make better
        for ind, val in enumerate(state['enemies']):
            obs = state['items']['obstacle'][self.covers[ind]]
            # Determine point to go to
            obs_pos = np.asarray([obs['x_position'], obs['y_position']])
            player_pos = np.asarray([state['player']['x_position'], state['player']['y_position']])

            angle = np.arctan2(obs_pos[0] - player_pos[0], obs_pos[1] - player_pos[1])

            goal = {'x_position': obs_pos[0] + -np.cos(angle) * cover_dist,
                    'y_position': obs_pos[1] + -np.sin(angle) * cover_dist}

            # Get info
            angle, sign = self.get_angle(goal, val)

            if angle < self.right_side:
                # Forward, left, right, shoot?
                action = random.choice([FORWARD, LEFT, RIGHT])
            else:
                if sign == -1.0:
                    # Turn right
                    action = TLEFT
                else:
                    # Turn left
                    action = TRIGHT

            # Send ai action
            commands.append([ind, action])

        return commands

    # Real novelty 203
    def teleport(self, state):
        # Function specific command
        teleport = 9

        commands = []

        # Update health table
        current_health = []
        for ind, val in enumerate(state['enemies']):
            current_health.append(val['health'])

        # Do logic
        for ind, val in enumerate(state['enemies']):
            # Check for double shots
            if current_health[ind] != self.last[ind]:
                action = teleport

            # Else use random (base environment) action
            else:
                action = random.choice([FORWARD, BACKWARD, LEFT, RIGHT, TLEFT, TRIGHT, SHOOT])

            commands.append([ind, action])

        # Update last tables
        self.lastlast = self.last
        self.last = current_health

        return commands

    # Real novelty 205
    def hunt(self, state):
        commands = []

        # Move to hunting
        if self.hunting:
            for ind, val in enumerate(state['enemies']):
                angle, sign = self.get_angle(state['player'], val)
                if angle < np.pi / 8:
                    action = SHOOT

                # Do movement here
                else:
                    if sign == -1.0:
                        # Turn right
                        action = TRIGHT
                    else:
                        # Turn left
                        action = TLEFT

                # Send ai action
                commands.append([ind, action])

        # Roll for the hunt
        else:
            if self.hunt_tick is None:
                r = np.random.rand()
                if self.difficulty == 'easy':
                    r = 10
                elif self.difficulty == 'medium':
                    r = 5
                elif self.difficulty == 'hard':
                    r = 0

                self.hunt_tick = r
            else:
                if self.tick_counter > self.hunt_tick:
                    self.hunting = True

            # Else use random (base environment) action
            for ind, val in enumerate(state['enemies']):
                action = random.choice([FORWARD, BACKWARD, LEFT, RIGHT, TLEFT, TRIGHT, SHOOT])
                commands.append([ind, action])

        return commands

    # Real novelty 207
    def point_defense(self, state):
        commands = []

        if self.difficulty == 'easy':
            point = np.asarray([256, -256])
            tolerance = 16
        elif self.difficulty == 'medium':
            point = np.asarray([0, 0])
            tolerance = 64
        elif self.difficulty == 'hard':
            point = np.asarray([-64, 128])
            tolerance = 128


        for ind, val in enumerate(state['enemies']):
            # Get info
            angle, sign = self.get_angle({'x_position': point[0], 'y_position': point[1]}, val)

            if angle < self.right_side:
                # Forward, left, right, shoot?
                action = random.choice([FORWARD, LEFT, RIGHT])
            else:
                if sign == -1.0:
                    action = TRIGHT
                else:
                    action = TLEFT

            # Send ai action
            commands.append([ind, action])

            # Check exit flag
            enemy_pos = np.asarray([val['x_position'], val['y_position']])
            if np.linalg.norm(enemy_pos - point) < tolerance:
                self.special_exit_flag = True

        return commands

    # Enemies shoot at player
    def check_shoot(self, state, commands):
        for ind, val in enumerate(state['enemies']):
            if commands[ind][1] == SHOOT or commands[ind][1] == 9:
                continue

            angle, sign = self.get_angle(state['player'], val)

            # Check enemy can shoot
            if angle < self.right_side:
                # Enemy has a 50% chance to shoot if aiming at player
                if np.random.rand() > 0.75:
                    commands[ind] = [ind, SHOOT]
                    # pass

        return commands

    # Enemies never shoot each other
    def check_enemies(self, state, commands):
        # From enemy
        for ind, val in enumerate(state['enemies']):
            # If an enemy is not shooting, dont mess with anything
            if commands[ind][1] != SHOOT:
                continue

            # To enemy
            for ind2, val2 in enumerate(state['enemies']):
                # Check for self
                if ind == ind2:
                    continue

                # Distance check sections
                # Get self angle
                angle = val['angle'] * 2 * np.pi / 360
                x = np.cos(angle)
                y = np.sin(angle)

                # Line start
                p1 = np.asarray([val['x_position'], val['y_position']])
                # Line end
                p2 = np.asarray([val['x_position'] + x, val['y_position'] + y])
                # Enemy point
                p3 = np.asarray([val2['x_position'], val2['y_position']])

                dist = np.abs(np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1))
                angle, sign = self.get_angle(val2, val)

                check_dist = min(self.last_dist[ind][ind2], dist)
                if check_dist < (30 + dist/20) and angle < np.pi / 2:
                    action = random.choice([1, 2, 3, 4])
                    commands[ind] = [ind, action]

                self.last_dist[ind][ind2] = dist

        return commands

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
            return np.random.rand() * 3.14, 1

        enemy_face_vector = np.asarray([pl_x - en_x, pl_y - en_y]) / np.linalg.norm(
            np.asarray([pl_x - en_x, pl_y - en_y]))

        angle = np.arccos(np.clip(np.dot(enemy_vector, enemy_face_vector), -1.0, 1.0))

        sign = np.sign(np.linalg.det(np.stack((enemy_vector[-2:], enemy_face_vector[-2:]))))

        return angle, sign

    def ccw(self, A, B, C):
        return (C['y']-A['y']) * (B['x']-A['x']) > (B['y']-A['y']) * (C['x']-A['x'])

    # Return true if line segments AB and CD intersect
    def intersect(self, A, B, C, D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

