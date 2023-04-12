"""
Vincent Lombardi
"""
from torch import nn
import torch
import numpy as np
import math


# 90 degree angle north


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_vec, buf_s, buf_a, buf_r, gamma):
    v = 0.0  # terminal
    if not done:
        v = lnet.forward(v_wrap(s_vec[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    i = len(buf_r) - 1
    # for reward in buf_r[::-1]:  # reverse buffer reward
    while i >= 0:
        v = buf_r[i] + gamma * v
        buffer_v_target.append(v)
        i -= 1
    buffer_v_target.reverse()
    # print(type(buf_a[0]))
    # if(buf_a[0].dtype != np.int64):
    #    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n\n")
    #    exit()

    # exit()
    loss = lnet.loss_func(
        v_wrap(np.vstack(buf_s)),
        v_wrap(np.array(buf_a), dtype=np.int64),  # if buf_a[0].dtype == np.int64 else v_wrap(np.vstack(buf_a)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


"""
def push_and_pull(opt, lnet, gnet, done, s_vec, buf_s, buf_a, buf_r, gamma):
    v = 0.  # terminal
    if not done:
        v = lnet.forward(v_wrap(s_vec[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for reward in buf_r[::-1]:  # reverse buffer reward
        v = reward + gamma * v
        buffer_v_target.append(v)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(buf_s)),
        v_wrap(np.array(buf_a), dtype=np.int64) if buf_a[0].dtype == np.int64 else v_wrap(np.vstack(buf_a)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())
"""


def record(global_ep, global_ep_r, ep_r, res_queue, name, enemies, kills, victory,
           dead, ammo, health, s_count):
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
        "s_count:", s_count,
        "| Ep_r: %.2f" % global_ep_r.value, " indiv: %.2f" % ep_r
    )


def record_comb(global_ep, global_ep_r, ep_r, res_queue, name, enemies, kills, victory,
                dead):
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
        "Ep:", global_ep.value, "enemies:", enemies, "kills:", kills, "victory:", victory, "dead", dead,
        "| Ep_r: %.2f" % global_ep_r.value, " indiv: %.2f" % ep_r
    )


def record_nav(global_ep, global_ep_r, ep_r, res_queue, name, dead, goal, t_coord, p_coord,
               gtype):
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
        "Ep:", global_ep.value, "goal:", goal, "t_coord:", t_coord, "p_coord:", p_coord,
        "| Ep_r: %.2f" % global_ep_r.value, " indiv: ", ep_r
    )


def in_center(e, p):
    if 60 > e['y_position'] > -60 and 55 > p['y_position'] > -55:
        return True
    elif 60 > e['x_position'] > -60 and 55 > p['x_position'] > -55:
        return True
    return False


def in_center2(p):
    if 50 > p['y_position'] > -50 or 50 > p['x_position'] > -50:
        return True
    return False


def in_center3(p):
    if 280 > p['y_position'] > -280 and 280 > p['x_position'] > -280:
        return True
    return False


def h_check(e):
    e_health = int(e['health'])
    if e_health == 10:
        # print("1")
        return 3
    elif e_health == 6:
        # print("2")
        return 2
    elif e_health == 2:
        # print("3")
        return 1
    else:
        # print("4")
        return 0


def corner(player):
    p_coord = tracker(player)
    if p_coord == 2:

        if player['x_position'] < 0:
            return 'left'
        else:
            return 'right'

    elif p_coord == 3:

        if player['x_position'] < 0:
            return 'right'
        else:
            return 'left'

    elif p_coord == 4:

        if player['y_position'] < 0:
            return 'right'
        else:
            return 'left'

    elif p_coord == 5:
        if player['y_position'] < 0:
            return 'left'
        else:
            return 'right'

    else:
        return 'nothing'


def door_man(player, target):
    p_coord = tracker(player)
    my_act = 'nothing'
    if target > 1 and p_coord == 1 and not in_center3(player):
        my_act = 'forward'
        if target == 2:
            if player['angle'] != 90.0:
                my_act = 'turn_left'

        elif target == 3:
            if player['angle'] != 270.0:
                my_act = 'turn_left'

        elif target == 4:
            if player['angle'] != 0.0:
                my_act = 'turn_left'

        elif target == 5:
            if player['angle'] != 180.0:
                my_act = 'turn_left'

    return my_act


def corner2(player, target):
    p_coord = target
    if p_coord == 2:

        if player['x_position'] < 0:
            return 'right'
        else:
            return 'left'

    elif p_coord == 3:

        if player['x_position'] < 0:
            return 'left'
        else:
            return 'right'

    elif p_coord == 4:

        if player['y_position'] < 0:
            return 'left'
        else:
            return 'right'

    elif p_coord == 5:
        if player['y_position'] < 0:
            return 'right'
        else:
            return 'left'

    else:

        return 'nothing'


def seeker(enemies, player):
    for e in enemies:
        if int(h_check(e)) <= int(player['ammo']):
            if target_sighted(e, player):
                return True
    return False


def gunner(e, p, offset, w=18):  # 18 guaranteed
    p_x = p['x_position']
    p_y = p['y_position']
    e_x = e['x_position']
    e_y = e['y_position']

    angle = p['angle'] + offset
    if angle < 0:
        angle = 360 + angle
    elif angle > 360:
        angle = angle - 360
    elif angle == 360:
        angle = 0
    if angle == 90 and e_y > p_y:  # north
        if (p_x - w) < e_x < (p_x + w):
            return True

    elif angle == 0 and e_x > p_x:  # east
        if (p_y - w) < e_y < (p_y + w):
            return True
    elif angle == 270 and e_y < p_y:  # south
        if (p_x - w) < e_x < (p_x + w):
            return True
    elif angle == 180 and e_x < p_x:  # west
        if (p_y - w) < e_y < (p_y + w):
            return True

    elif angle == 45 and e_x > p_x and e_y > p_y:
        dif = e_x - p_x
        if (p_y + dif - w) < e_y < (p_y + dif + w):
            return True
    elif angle == 225 and e_x < p_x and e_y < p_y:
        dif = p_x - e_x
        if (p_y - dif - w) < e_y < (p_y - dif + w):
            return True

    elif angle == 135 and e_x < p_x and e_y > p_y:
        dif = p_x - e_x

        if (p_y + dif - w) < e_y < (p_y + dif + w):
            return True

    elif angle == 315 and e_x > p_x and e_y < p_y:
        dif = e_x - p_x

        if (p_y - dif - w) < e_y < (p_y - dif + w):
            return True

    return False


def target_sighted(e, p):
    p_x = p['x_position']
    p_y = p['y_position']
    e_x = e['x_position']
    e_y = e['y_position']
    if e_y > 352 and p_y > 352:
        return True
    elif e_y < -352 and p_y < -352:
        return True
    elif e_x < -352 and p_x < -352:
        return True
    elif e_x > 352 and p_x > 352:
        return True
    elif 352 > e_y > -352 and 352 > e_x > -352:
        if 352 > p_y > -352 and 352 > p_x > -352:
            return True
    return in_center(e, p)


def in_door(e, p):
    angle = p['angle']

    if tracker2(e) != 6 and tracker2(p) != 6:
        return True

    elif tracker2(e) == 6 or tracker2(p) == 6:
        if angle == 0.0 or angle == 90.0 or angle == 180 or angle == 270:
            return True
    return False


def tracker(t):
    coord = tracker2(t)
    if coord < 6:
        return coord
    else:
        return 1  # 6#"d"


def tracker2(t):
    t_x = t['x_position']
    t_y = t['y_position']
    if 320 > t_y > -320 and 320 > t_x > -320:  # central
        return 1  # "c"
    elif t_y > 384:  # north
        return 2  # "n"
    elif t_y < -384:  # south
        return 3  # "s"
    elif t_x > 384:  # east
        return 4  # "e"
    elif t_x < -384:  # west
        return 5  # "w"
    else:
        return 6  # "d"


def target_in_room(enemies, p_coord):
    if p_coord == 6:
        return False

    for e in enemies:
        e_coor = tracker(e)
        if p_coord == e_coor:
            return True
    return False


def navigate(enemies, p_coord, player):
    if p_coord != 1:
        return 1

    if len(enemies) <= 0:
        return 0
    min_dist = 10000
    target = None
    for e in enemies:
        dist = get_dist(player, e)
        if min_dist > dist:
            min_dist = dist
            target = e

    return tracker(target)


def get_dist(player, enemy):
    pl_x = player['x_position']
    pl_y = player['y_position']

    en_x = enemy['x_position']
    en_y = enemy['y_position']
    e_coor = [en_x, en_y]
    p_coor = [pl_x, pl_y]

    return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p_coor, e_coor)))  # math.dist(p_coor, e_coor)


# Utility function for getting angle from B-direction to A


def get_angle(target, start, offset):
    pl_x = target['x_position']
    pl_y = target['y_position']

    en_x = start['x_position']
    en_y = start['y_position']
    ang = start['angle'] + offset
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

    return angle
