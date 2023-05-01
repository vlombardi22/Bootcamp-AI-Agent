import numpy as np
import math


def record(global_ep, global_ep_r, ep_r, res_queue, name, enemies, kills, victory,
           dead, ammo, dist):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)

    print(
        name,
        "Ep:", global_ep.value, "enemies:", enemies, "kills:", kills, "victory:", victory, "dead:", dead, "ammo:", ammo,
        "| Ep_r: %.2f" % global_ep_r.value, " indiv: %.2f" % ep_r, "dist:", dist
    )


def record_fell(global_ep, global_ep_r, ep_r, res_queue, name, enemies, kills, victory,
                dead, ammo, task_var, p_queue, f_queue):
    test = False
    test2 = False
    with global_ep.get_lock():
        global_ep.value += 1
        if global_ep.value < 200:
            test = True
        if global_ep.value >= 800:
            test2 = True

    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01

    if test:
        p_queue.put(ep_r)
    if test2:
        f_queue.put(ep_r)

    res_queue.put(global_ep_r.value)
    task = "combat"

    if task_var == 2:
        task = "reload"
    elif task_var == 3:
        task = "heal"

    print(
        name,
        "Ep:", global_ep.value, "enemies:", enemies, "kills:", kills, "victory:", victory, "dead:", dead, "ammo:", ammo,
        "| Ep_r: %.2f" % global_ep_r.value, " indiv: %.2f" % ep_r, task
    )


def record_dead(global_ep, global_ep_r, ep_r, res_queue, name, enemies, kills, victory,
                dead, ammo, p_queue, dist, step, tk, a_queue):
    test = False
    test2 = False
    with global_ep.get_lock():
        global_ep.value += 1
        if global_ep.value < 200:
            test = True
        if global_ep.value >= 800:
            test2 = True

    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01

    if test:
        p_queue.put(ep_r)
    if test2:
        a_queue.put(ep_r)
    res_queue.put(global_ep_r.value)

    print(
        name,
        "Ep:", global_ep.value, "enemies:", enemies, "kills:", kills, "victory:", victory, "dead:", dead,
        "| Ep_r: %.2f" % global_ep_r.value, " indiv: %.2f" % ep_r, "step:", step, "dist:", dist
    )


def get_dist(player, enemy):
    pl_x = player.position_x
    pl_y = player.position_y

    en_x = enemy.position_x
    en_y = enemy.position_y
    e_coor = [en_x, en_y]
    p_coor = [pl_x, pl_y]

    return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p_coor, e_coor)))


def get_angle(target, start, offset=0.0):
    pl_x = target.position_x
    pl_y = target.position_y

    en_x = start.position_x
    en_y = start.position_y

    ang = round(start.angle, 4) + offset  # * 2 * np.pi / 360 + offset
    if ang < 0:
        ang = 360 + ang
    elif ang > 360:
        ang = ang - 360
    elif ang == 360:
        ang = 0

    en_ori = ang * 2 * np.pi / 360

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


def gunner(e, p, offset=0.0, w=5, goal=False):  # 18 guaranteed
    angle, _ = get_angle(e, p, offset)

    angle = angle * 180 / np.pi

    if angle < w and (get_dist(p, e) < 250 or goal):
        return 2

    return 7


def move(target, player):
    a, sign = get_angle(target, player)

    if sign < 0:
        return 6

    else:
        return 5


def get_armor(armor, player):
    if gunner(armor, player, w=10, goal=True) == 2:
        return 3
    else:
        return move(armor, player)


def get_ammo(ammo, player):
    if get_dist(ammo, player) < 15:
        t = [0, 1, 3, 4]
        return t[np.random.randint(0, 4)]
    else:
        return get_armor(ammo, player)


def navigate(target, player):
    if not (-32.0 < target.position_y < 32.0):
        if -32.0 > player.position_y:
            return 0
        elif player.position_y > 32.0:
            return 1

    return get_armor(target, player)


def fight(target, player):
    if gunner(target, player) == 2:
        return 2
    return move(target, player)


def get_gun(player, weapon):
    return get_armor(weapon, player)


def charge(player, enemy):
    return get_armor(enemy, player)
