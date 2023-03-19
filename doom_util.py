"""
Vincent Lombardi
"""
from torch import nn
import torch
import numpy as np
import math
import csv

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

def record_base(global_ep, global_ep_r, ep_r, res_queue, name, enemies, kills, victory,
                dead, ammo, health, task_var):
    with global_ep.get_lock():
        global_ep.value += 1

    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01

    res_queue.put(global_ep_r.value)
    task = "combat"

    if task_var == 2:
        task = "reload"
    elif task_var == 3:
        task = "heal"

    print(
        name,
        "Ep:", global_ep.value, "enemies:", enemies, "kills:", kills, "victory:", victory, "dead:", dead, "ammo:", ammo,
        "health:", health,
        "| Ep_r: %.2f" % global_ep_r.value, " indiv: %.2f" % ep_r, task
    )

def record_fell_ppo(global_ep, global_ep_r, ep_r, res_queue, enemies, kills, victory,
                dead, ammo, health, task_var, p_queue, f_queue):
    test = False
    test2 = False
    with global_ep.get_lock():
        global_ep.value += 1
        if global_ep.value < 20:
            test = True
        if global_ep.value >= 80:
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
        "Ep:", global_ep.value, "enemies:", enemies, "kills:", kills, "victory:", victory, "dead:", dead, "ammo:", ammo,
        "health:", health,
        "| Ep_r: %.2f" % global_ep_r.value, " indiv: %.2f" % ep_r, task
    )



def record_fell(global_ep, global_ep_r, ep_r, res_queue, name, enemies, kills, victory,
                dead, ammo, health, task_var, p_queue, f_queue):
    test = False
    test2 = False
    with global_ep.get_lock():
        global_ep.value += 1
        if global_ep.value < 200:
            test = True
        if global_ep.value >= 800:
            test2 = True
        # if global_ep.value >= 5:#500:
        #    test2

    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01

    if test:
        p_queue.put(ep_r)
    if test2:
        f_queue.put(ep_r)
        #f2_queue.put(ep_rr)

    # if test2:
    #    my_p2.put(ep_r)
    # my_p2.put(ep_r)
    res_queue.put(global_ep_r.value)
    task = "combat"

    if task_var == 2:
        task = "reload"
    elif task_var == 3:
        task = "heal"

    print(
        name,
        "Ep:", global_ep.value, "enemies:", enemies, "kills:", kills, "victory:", victory, "dead:", dead, "ammo:", ammo,
        "health:", health,
        "| Ep_r: %.2f" % global_ep_r.value, " indiv: %.2f" % ep_r, task
    )

def record_boot(global_ep, global_ep_r, ep_r, res_queue, name, enemies, kills, victory,
                dead, ammo, health, s_count, task_var, global_kills, global_health, global_ammo, MAX_EP):
    test = False
    with global_ep.get_lock():
        global_ep.value += 1
        if global_ep.value > MAX_EP - 500:
            test = True
        if MAX_EP <= 500:
            test = True
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    if test:
        with global_kills.get_lock():
            global_kills.value += kills
        with global_ammo.get_lock():
            global_ammo.value += ammo
        with global_health.get_lock():
            global_health.value += health
    res_queue.put(global_ep_r.value)
    task = "combat"

    if task_var == 2:
        task = "reload"
    elif task_var == 3:
        task = "heal"

    print(
        name,
        "Ep:", global_ep.value, "enemies:", enemies, "kills:", kills, "victory:", victory, "dead:", dead, "ammo:", ammo,
        "health:", health,
        "| Ep_r: %.2f" % global_ep_r.value, " indiv: %.2f" % ep_r, task
    )





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

    if 60 > e_y > -60 and 55 > p_y > -55:
        return True
    elif 60 > e_x > -60 and 55 > p_x > -55:
        return True
    return False  # in_center(e, p)


def in_door(e, p):
    angle = p['angle']
    tp = tracker2(p)
    te = tracker2(e)

    if te != 6 and tp != 6:
        return True

    elif te == 6 or tp == 6:
        if angle == 0.0 or angle == 90.0 or angle == 180 or angle == 270:
            return True
    return False


def tracker(t):
    """
    coord = tracker2(t)
    if coord < 6:
        return coord
    else:
        return 1  # 6#"d"
    """
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
        return 1  # "d"


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
    sign = np.sign(np.linalg.det(np.stack((enemy_vector[-2:], enemy_face_vector[-2:]))))
    return angle, sign


def break_obstacles(test_obst, player):  # items, player):
    # nav_obst = []
    ob_list = []
    min_dist = 10000
    m_obst = None

    for o in test_obst:  # items['obstacle']:
        dist = get_dist(player, o)

        if target_sighted(o, player):
            ob_list.append(o)

        if min_dist > dist:
            min_dist = dist
            m_obst = o

    if not m_obst:  # len(items['obstacle']) <= 0:
        nav_obst = [0.0, 0.0, 0.0]

    else:
        nav_obst = [float(m_obst['x_position']), float(m_obst['y_position']), min_dist]
    return nav_obst, ob_list


def break_traps(items, player):
    if len(items['trap']) <= 0:
        return [0.0, 0.0, 0.0]

    min_dist = 10000
    m_trap = items['trap'][0]
    for t in items['trap']:
        dist = get_dist(player, t)
        if min_dist > dist:
            min_dist = dist
            m_trap = t
    nav_trap = [float(m_trap['x_position']), float(m_trap['y_position']), min_dist]

    return nav_trap


def break_ammo(items, player, p_coord, enemies):
    # clip_list = items['ammo']
    if len(items['ammo']) <= 0:
        return None, [0.0, 0.0, -1.0, 0.0]
    min_dist = 10000
    #strat_clip = []
    m_ammo = None
    m_back = None
    a_coord = 0  # tracker(m_ammo)
    """
    if empty
    (elif current and prospective target not in room go to min
        elif current target not in room but prospective target is in room go to prospective)
    elif current and prospective target in room go to min
    elif prospective target == 1 mark as backup
    """
    for a in items['ammo']:
        dist = get_dist(player, a)
        t_coord = tracker(a)
        if not m_ammo:  # if empty
            min_dist = dist
            m_ammo = a
            a_coord = t_coord
        elif p_coord != a_coord and ((p_coord != t_coord and min_dist > dist) or p_coord == t_coord):
            min_dist = dist
            m_ammo = a
            a_coord = t_coord
        elif p_coord == a_coord and p_coord == t_coord and min_dist > dist:
            min_dist = dist
            m_ammo = a
            a_coord = t_coord
        elif tracker(a) == 1:
            m_back = a

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if len(enemies) > 0:
        """
        a_coord = tracker(m_ammo)
        if a_coord != p_coord:
            for a in clip_list:
                a_c = tracker(a)
                if a_c == p_coord:
                    m_ammo = a
                    a_coord = a_c
                    break
        """
        if a_coord != 1 and p_coord == a_coord and m_back:
            hazard = False
            for e in enemies:  # bookmarke
                if tracker(e) == a_coord:
                    hazard = True
                    break
            if hazard:
                m_ammo = m_back
    """
    strat_clip.append(float(m_ammo['x_position']))
    strat_clip.append(float(m_ammo['y_position']))
    strat_clip.append(get_dist(player, m_ammo))
    angle, sign = get_angle(m_ammo, player, 0.0)
    angle = angle * 180 / np.pi
    strat_clip.append(angle)
    """
    angle, _ = get_angle(m_ammo, player, 0.0)
    angle = angle * 180 / np.pi
    strat_clip = [float(m_ammo['x_position']), float(m_ammo['y_position']), get_dist(player, m_ammo), angle]
    return m_ammo, strat_clip


def break_health(items, player, p_coord, enemies):
    if len(items['health']) <= 0:
        return None, [0.0, 0.0, -1.0, 0.0]
    min_dist = 10000
    m_back = None
    m_health = None
    # strat_health = []
    m_coord = 0
    # med_list = items['health']
    for h in items['health']:  # med_list:
        dist = get_dist(player, h)
        t_coord = tracker(h)
        if not m_health:  # if empty
            min_dist = dist
            m_health = h
            m_coord = t_coord
        elif p_coord != m_coord and ((p_coord != t_coord and min_dist > dist) or p_coord == t_coord):
            min_dist = dist
            m_health = h
            m_coord = t_coord
        elif p_coord == m_coord and p_coord == t_coord and min_dist > dist:
            min_dist = dist
            m_health = h
            m_coord = t_coord
        elif tracker(h) == 1:
            m_back = h
    """
    m_coord = tracker(m_health)
    if m_coord != p_coord:
        for h in med_list:
            h_c = tracker(h)
            if h_c == p_coord:
                m_health = h
                m_coord = h_c
                break
    """
    if m_coord != 1 and p_coord == m_coord and m_back:
        hazard = False

        for e in enemies:  # bookmarke
            if tracker(e) == m_coord:
                hazard = True
                break

        if hazard:
            m_health = m_back
            # for h in med_list:
            #    if tracker(h) == 1:
            #        m_health = h
            #        break
    # strat_health.append(float(m_health['x_position']))
    # strat_health.append(float(m_health['y_position']))
    # strat_health.append(get_dist(player, m_health))
    angle, _ = get_angle(m_health, player, 0.0)
    angle = angle * 180 / np.pi
    # strat_health.append(angle)
    strat_health = [float(m_health['x_position']), float(m_health['y_position']), get_dist(player, m_health), angle]

    return m_health, strat_health


def break_enemy2(player, ob_list, enemies, p_coord):
    a_1 = True
    a_2 = True
    a_3 = True
    a_4 = True
    seek = False

    nav_enemy = []
    strat_enemy = []
    combat = 0  # 0 nothing, 1 attack, 2 left, 3 right, 4 m left, 5 m right
    w_0 = 0
    min_dist = 10000
    min_dist2 = 10000
    m_enemy = None
    targ_enemy = None
    m_targ = None
    ammo = player['ammo']
    firing = False  # do we have a clear line of fire
    # targ_coord = 0

    # if len(enemies) <= 0:
    #    #return 0
    #    targ_coord = 0
    tir = False

    if tracker2(player) != 6:
        w_0 = 20
    for e in enemies:  # bookmarke
        dist = get_dist(player, e)
        ts = target_sighted(e, player)
        if p_coord != 6 and p_coord == tracker(e):
            tir = True
        if min_dist2 > dist:
            min_dist2 = dist
            targ_enemy = e

        if ts:
            if min_dist > dist:
                min_dist = dist
                m_enemy = e
            if ammo > 0 and not firing:
                # if ts:
                if int(h_check(e)) <= ammo:
                    seek = True

                if in_door(e, player):
                    a_0 = False

                    # dist = get_dist(player, e)
                    if gunner(e, player, 0.0, 40 + w_0):

                        barr = ob_help(ob_list, player, dist, 0.0)
                        is_clear = True
                        if barr:
                            if gunner(e, player, 0.0, 18):
                                combat = 1
                                # strat_enemy = helper(e, player)
                                m_targ = e
                                firing = True
                                # break
                        elif is_clear:
                            ang, sign = get_angle(e, player, 0.0)

                            check3 = True
                            check4 = False
                            check5 = True
                            for o in ob_list:

                                if get_dist(player, o) < 60:
                                    check4 = True
                                    ang2, sign2 = get_angle(o, player, 0.0)
                                    ang2 = ang2 * 180 / np.pi
                                    if 135 > ang2 > 45:
                                        if sign2 < 1:

                                            check3 = False
                                        else:
                                            check5 = False

                            if check3 or check5:

                                if sign < 1 and check3:
                                    # print("right")
                                    combat = 5
                                    if not check4:
                                        a_0 = True  # we don't want to repeat
                                        a_2 = True
                                elif sign == 1 and check5:
                                    # print("left")
                                    combat = 4
                                    if not check4:
                                        a_0 = True  # we don't want to repeat
                                        a_2 = True
                    if a_1 and not firing:
                        w_1 = 30 + w_0
                        if a_0:
                            w_1 = 18

                        if gunner(e, player, 45.0, w_1) and ob_help(ob_list, player, dist, 45.0):
                            combat = 2
                            a_1 = False
                            # strat_enemy = helper(e, player)
                            m_targ = e

                        if a_1 and gunner(e, player, -45.0, w_1) and ob_help(ob_list, player, dist, -45.0):
                            combat = 3
                            a_1 = False
                            # strat_enemy = helper(e, player)
                            m_targ = e
                        if a_1 and a_2 and gunner(e, player, 90.0, 30 + w_0) and ob_help(ob_list, player, dist, 90.0):
                            combat = 2
                            a_2 = False
                            # strat_enemy = helper(e, player)
                            m_targ = e
                        if a_1 and a_2 and gunner(e, player, -90.0, 30 + w_0) and ob_help(ob_list, player, dist, -90.0):
                            combat = 3
                            a_2 = False
                            # strat_enemy = helper(e, player)
                            m_targ = e
                        if a_1 and a_2 and a_3 and gunner(e, player, 135.0, 30 + w_0) and ob_help(ob_list, player, dist,
                                                                                                  135.0):
                            combat = 2
                            a_3 = False
                            # strat_enemy = helper(e, player)
                            m_targ = e
                        if a_1 and a_2 and a_3 and gunner(e, player, -135.0, 30 + w_0) and ob_help(ob_list, player,
                                                                                                   dist,
                                                                                                   -135.0):
                            combat = 3
                            a_3 = False
                            # strat_enemy = helper(e, player)
                            m_targ = e
                        if a_1 and a_2 and a_3 and a_4 and gunner(e, player, 180.0, 30 + w_0) and ob_help(ob_list,
                                                                                                          player,
                                                                                                          dist, 180.0):
                            combat = 2
                            a_4 = False
                            # strat_enemy = helper(e, player)
                            m_targ = e

    if p_coord != 1:
        targ_coord = 1
    elif not targ_enemy:
        targ_coord = 0
    else:
        targ_coord = tracker(targ_enemy)

    if not m_enemy:
        nav_enemy = [0.0, 0.0, 10000.0]
        strat_enemy = [0.0, 0.0, 0.0, -1.0, 0.0]
        # targ_coord = 0
    else:
        nav_enemy = [float(m_enemy['x_position']), float(m_enemy['y_position']), min_dist]
        # if p_coord == 1:
        #    targ_coord = tracker(m_enemy)

        if m_targ:
            # strat_enemy = helper(m_targ, player)
            angle, _ = get_angle(m_targ, player, 0.0)
            angle = angle * 180 / np.pi
            strat_enemy = [m_targ['x_position'], m_targ['y_position'], h_check(m_targ), get_dist(m_targ, player), angle]
        else:
            # strat_enemy = helper(m_enemy, player)
            angle, _ = get_angle(m_enemy, player, 0.0)
            angle = angle * 180 / np.pi
            strat_enemy = [m_enemy['x_position'], m_enemy['y_position'], h_check(m_enemy), get_dist(m_enemy, player),
                           angle]
    return nav_enemy, strat_enemy, combat, targ_coord, tir, seek


def breaker(state, test_obst):  # bookmark
    enemies = state['enemies']
    items = state['items']
    player = state['player']

    # combat = 0  # 0 nothing, 1 attack, 2 left, 3 right, 4 m left, 5 m right
    nav_obst, ob_list = break_obstacles(test_obst, player)  # items, player)
    p_coord = tracker(player)

    nav_enemy, strat_enemy, combat, targ_coord, tir, seek = break_enemy2(player, ob_list, enemies, p_coord)
    #print(len(strat_enemy))
    #exit()
    e_count = len(enemies)
    a_count = len(items['ammo'])
    h_count = len(items['health'])
    # nav_trap = break_traps(items, player)

    clip, strat_ammo = break_ammo(items, player, p_coord, enemies)
    med, strat_health = break_health(items, player, p_coord, enemies)
    sensor_vec = [float(player['x_position']), float(player['y_position']), float(player['angle']), int(player['ammo']),
                  int(player['health']), e_count] + strat_enemy + [a_count] + strat_ammo + [h_count] + strat_health + [
                     0.0, 0.0, 0.0, 0.0]

    if nav_enemy[2] < 90.0 and (nav_enemy[2] < nav_obst[2] or nav_obst[2] <= 0):
        nav_obst = nav_enemy
    avatar2 = [float(player['x_position']), float(player['y_position']), float(player['angle']), 30]  # , 20]

    sensor_vec2 = avatar2 + [0.0, 0.0, 0.0] + nav_obst + [0.0, 0.0, -1.0]
    return np.asarray(sensor_vec), np.asarray(
        sensor_vec2), e_count, combat, clip, med, targ_coord, tir, seek


def to_border(player, target):
    my_act = np.dtype('int64').type(3)
    angle = player['angle']
    if target > 1:
        if target == 2:
            if angle != 90.0:
                if 270 > angle > 90:
                    my_act = np.dtype('int64').type(5)  # 'turn_right'
                else:
                    my_act = np.dtype('int64').type(4)

        elif target == 3:
            if angle != 270.0:
                if 270 > angle > 90:
                    my_act = np.dtype('int64').type(4)  # 'turn_left'
                else:
                    my_act = np.dtype('int64').type(5)
        elif target == 4:
            if angle != 0.0:
                if angle > 180:
                    my_act = np.dtype('int64').type(4)  # 'turn_left'
                else:
                    my_act = np.dtype('int64').type(5)

        elif target == 5:
            if angle != 180.0:
                if angle > 180:
                    my_act = np.dtype('int64').type(5)  # 'turn_right'
                else:
                    my_act = np.dtype('int64').type(4)

        if my_act == 3 and not (
                in_center2(player)):  # or (tracker2(player) == 6 and get_dist(player, self.c_list[target-1]) > 200)):
            if target == 2:

                if player['x_position'] < 0:
                    my_act = np.dtype('int64').type(1)
                else:
                    my_act = np.dtype('int64').type(0)

            elif target == 3:

                if player['x_position'] < 0:
                    my_act = np.dtype('int64').type(0)
                else:
                    my_act = np.dtype('int64').type(1)

            elif target == 4:

                if player['y_position'] < 0:
                    my_act = np.dtype('int64').type(0)
                else:
                    my_act = np.dtype('int64').type(1)

            elif target == 5:
                if player['y_position'] < 0:
                    my_act = np.dtype('int64').type(1)
                else:
                    my_act = np.dtype('int64').type(0)

    return my_act


def ob_help(ob_list, player, e_dist, offset):
    for o in ob_list:
        w = 40.0
        if tracker(o) != 1:
            w = 25.0
        if gunner(o, player, offset, w):
            if e_dist > get_dist(player, o):
                return False
    return True


def to_center(player, p_coord):
    my_act = np.dtype('int64').type(3)  # 'nothing'  # 'forward'
    angle = player['angle']
    if p_coord == 2:
        if angle != 270.0:
            if 270 > angle > 90:
                my_act = np.dtype('int64').type(4)  # 'turn_left'
            else:
                my_act = np.dtype('int64').type(5)

    elif p_coord == 3:
        if angle != 90.0:
            if 270 > angle > 90:
                my_act = np.dtype('int64').type(5)  # 'turn_right'
            else:
                my_act = np.dtype('int64').type(4)
    elif p_coord == 4:
        if angle != 180.0:
            if angle > 180:
                my_act = np.dtype('int64').type(5)  # 'turn_right'
            else:
                my_act = np.dtype('int64').type(4)

    elif p_coord == 5:
        if angle != 0.0:
            if angle > 180:
                my_act = np.dtype('int64').type(4)  # 'turn_left'
            else:
                my_act = np.dtype('int64').type(5)

    if my_act == 3:  # 'nothing':
        if not in_center2(player):
            if p_coord == 2:

                if player['x_position'] < 0:
                    my_act = np.dtype('int64').type(0)  # 'left'
                else:
                    my_act = np.dtype('int64').type(1)  # 'right'

            elif p_coord == 3:

                if player['x_position'] < 0:
                    my_act = np.dtype('int64').type(1)  # 'right'
                else:
                    my_act = np.dtype('int64').type(0)  # 'left'

            elif p_coord == 4:

                if player['y_position'] < 0:
                    my_act = np.dtype('int64').type(1)  # 'right'
                else:
                    my_act = np.dtype('int64').type(0)  # 'left'

            elif p_coord == 5:
                if player['y_position'] < 0:
                    my_act = np.dtype('int64').type(0)  # 'left'
                else:
                    my_act = np.dtype('int64').type(1)  # 'right'

    return my_act