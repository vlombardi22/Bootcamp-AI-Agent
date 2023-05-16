import numpy as np

def action_trans(action):
    """
    turns actions into label
    :param action:
    :return:
    """
    action_name = ""
    if action == 0:
        action_name = 'left'
    elif action == 1:
        action_name = 'right'
    elif action == 2:
        action_name = 'backward'
    elif action == 3:
        action_name = 'forward'
    elif action == 4:
        action_name = 'turn_left'
    elif action == 5:
        action_name = 'turn_right'
    elif action == 6:
        action_name = 'shoot'
    return action_name


def get_angle(player, enemy):
    """
    get angle
    :param player:
    :param enemy:
    :return:
    """
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
        return np.random.rand() * 3.14

    enemy_face_vector = np.asarray([pl_x - en_x, pl_y - en_y]) / np.linalg.norm(
        np.asarray([pl_x - en_x, pl_y - en_y]))

    angle = np.arccos(np.clip(np.dot(enemy_vector, enemy_face_vector), -1.0, 1.0))

    sign = np.sign(np.linalg.det(
        np.stack((enemy_vector[-2:], enemy_face_vector[-2:]))
    ))

    return angle, sign


def ccw(A, B, C):
    return (C['y'] - A['y']) * (B['x'] - A['x']) > (B['y'] - A['y']) * (C['x'] - A['x'])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

