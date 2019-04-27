
import arrow
import collections
from generals import *
import numpy as np
from scipy.signal import convolve2d
import helpers


INF = 999999
DIRECTIONS = [
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
]


# TODO fix explore if not seen by turn 50
# TODO make attack more dfs


def start_game(region, userid, username, mode, gameid):
    gens2 = set()
    seen = False
    attack_this_round = False
    client = Generals(userid, username, mode, gameid=gameid,
                      region=region)
    numcities2 = 0
    prevdarmies = 0
    for state in client.get_updates():
        print
        print "Time", arrow.now()

        # print result if game is finished
        if state['complete']:
            print ["lose", "win"][state['result']], \
                "v", state['usernames'][1-pi].encode('utf-8'), \
                "[{}]".format(state['stars'][1-pi]), \
                state['replay_url']
            break

        # print scoreboard
        print "Turn {}".format(state['turn']/2)
        for u, star, land, army in zip(state['usernames'], state['stars'],
                                       state['lands'], state['armies']):
            print "{:<25}Army: {:<5} Land: {}".format(
                u.encode('utf-8') + " [{}]".format(star), army, land)

        moves = helpers.get_all_moves(state)

        pi = state['player_index']
        tg = state['tile_grid'] = np.asarray(state['tile_grid'])
        ag = state['army_grid'] = np.asarray(state['army_grid'])

        # only look at 200 biggest moves
        moves = sorted(moves, key=lambda (ay, ax, by, bx): -ag[ay, ax])[:200]
        print (tg == pi) * ag

        gen1 = state['generals'][pi]

        ply = state['turn']
        turn = state['turn'] / 2

        # resign
        if (turn > 1000 and len(_ for l in state['lands'] if l > 0) == 2
                and sum(state['lands']) > 11 * state['lands'][pi]
                and sum(state['armies']) > 11 * state['armies'][pi]):
            print 'resign'
            break

        p1 = tg == pi
        p2 = (tg >= 0) & (tg != pi)

        city = np.zeros(tg.shape).astype(bool)
        numcities1 = 0
        for y, x in state['cities']:
            if tg[y, x] != pi:
                city[y, x] = True
            else:
                numcities1 += 1
            if (y, x) in gens2:
                gens2.discard((y, x))
        empty = (tg == EMPTY) & ~city
        obstacle = (tg == MOUNTAIN) | (tg == OBSTACLE)

        # calculate initial distances
        if not seen:
            seen = True
            distgen1 = bfs([gen1], state)
            fargen1 = maxwhere(distgen1, distgen1 < INF)
            distfargen1 = bfs(fargen1[:1], state)
            explored = np.zeros(tg.shape).astype(bool)
            ochoke = np.zeros(tg.shape).astype(bool)
            ichoke = np.zeros(tg.shape).astype(bool)
            ichoke[gen1[0], gen1[1]] = True
            for y in range(state['rows']):
                for x in range(state['cols']):
                    ocount = icount = 0
                    for dy, dx in helpers.DIRECTIONS:
                        by, bx = y+dy, x+dx
                        if (0 <= by < state['rows'] and 0 <= bx < state['cols']
                                and tg[by][bx] not in [MOUNTAIN, OBSTACLE]):
                            if distgen1[by, bx] == distgen1[y, x] - 1:
                                ocount += 1
                            elif distgen1[by, bx] == distgen1[y, x] + 1:
                                icount += 1
                    if ocount <= 1:
                        ochoke[y, x] = True
                    if icount <= 1:
                        ichoke[y, x] = True

            dgmean = np.mean(distgen1[distgen1 < INF])
            if dgmean < 12:
                print 'central spawn'
                takecities = False
            elif dgmean < 15:
                print 'average spawn'
                takecities = False
            else:
                print 'far spawn'
                takecities = False

        explored |= tg != FOG
        distexplored = bfs(where(explored) + [gen1], state)
        farexplored = maxwhere(distexplored, distexplored < INF)
        distfarexplored = bfs(farexplored[:1], state)

        # update distances every ply
        distp1 = bfs(where(p1), state)
        farp1 = maxwhere(distp1, distp1 < INF)
        distfarp1 = bfs(farp1[:1], state)
        distp2 = bfs(where(p2), state)

        # if city.sum():
        #     nearcity1 = minwhere(distgen1, city)
        #     distnearcity1 = bfs(nearcity1[:1], state)

        # update opponent generals
        for i, (y, x) in enumerate(state['generals']):
            if y >= 0 and x >= 0 and i != pi:
                gens2.add((y, x))
            if y >= 0 and x >= 0 and tg[y, x] == pi:
                gens2.discard((y, x))

        # calculate threats
        threat = p2 * np.maximum(0, ag - 2 * distgen1
                                 - ag[gen1[0], gen1[1]] - 1)
        if gens2:
            distgens2 = bfs(gens2, state)
            threat *= distgen1 < distgens2
        else:
            distgens2 = None
            threat *= distgen1 <= 16

        isthreat = False
        if threat.sum() > 0:
            ty, tx = minwhere(distgen1, threat > 0)[0]
            threatpath = get_path(ty, tx, distgen1)
            threatpath[ty, tx] = False
            n = distgen1[ty, tx]
            distnearthreat = bfs([(ty, tx)], state, filt=(distgen1 <= n))

            a = (distgen1 < n - 1) | (distgen1 == 0)
            b = (distgen1 <= n - 1) & (distgen1 + n > distnearthreat)
            close = a | b
            close2 = close | adj(close)
            if threat[ty, tx] <= (p1 * ag)[close2].sum():
                isthreat = True
                print "defend"
            else:
                print "can't defend"

        safecity = city * (distp2 >= 5)
        if safecity.sum():
            distsafecity = bfs(maxwhere(distp2, safecity > 0)[:1], state)

        darmies = 2 * state['armies'][pi] - sum(state['armies'])
        generated = (ply % 50 == 0) * (
            2 * state['lands'][pi] - sum(state['lands']))
        if darmies > prevdarmies + 30 + generated:
            numcities2 += 1
            print 'numcities2', numcities2
        prevdarmies = darmies

        if not moves:
            move = None

        # opening (turns 0 - 25)
        elif turn < 25:
            num_chokes = min((distgen1 == i).sum() for i in range(3, 10))
            launch = [24, 36] if num_chokes > 1 else [28]

            # move from general
            if ply in launch:
                print "launch"

                # explore but keep liberties around general
                def f(move):
                    s = helpers.apply_move(state, move)
                    ag = np.asarray(s['army_grid'])
                    tg = np.asarray(s['tile_grid'])
                    p1 = tg == pi
                    explore = (distfarp1 * p1 * ag).sum() / (p1 * ag).sum()
                    return s['armies'][pi] - explore - 10*s['lands'][pi]

                gen1moves = helpers.get_moves(state, gen1[0], gen1[1])
                move = max(gen1moves, key=f) if gen1moves else None

            # greedily grab land around general
            elif turn >= 21:
                print "greedy"

                # grab land or move towards nearest empty to general
                def f(move):
                    s = helpers.apply_move(state, move)
                    ag = np.asarray(s['army_grid'])
                    tg = np.asarray(s['tile_grid'])
                    dland = 2 * s['lands'][pi] - sum(s['lands'])
                    p1 = tg == pi
                    empty = (tg == EMPTY) & ~city
                    genliberties = ((distgen1 <= 1) & empty).sum()
                    distempty = bfs(minwhere(distgen1, empty), s)
                    x = (distempty * p1 * ag).sum() / (p1 * ag).sum()
                    return (s['armies'][pi] + 10*dland - x
                            + 0.01*genliberties)

                move = max(moves, key=f)

            # explore
            else:
                print "explore"

                # explore but keep liberties around general
                def f(move):
                    s = helpers.apply_move(state, move)
                    ag = np.asarray(s['army_grid'])
                    tg = np.asarray(s['tile_grid'])
                    p1 = tg == pi
                    explore = (distfargen1 * p1 * ag).sum() / (p1 * ag).sum()
                    distp1 = bfs(where(p1), s)
                    empty = (tg == EMPTY) & ~city
                    genliberties = ((distgen1 <= 1) & empty).sum()
                    # takecity = -(city * (~p1) * (tg >= 0)).sum()
                    takecity = -(city * p2).sum()
                    return (s['armies'][pi] + 10*s['lands'][pi] - explore
                            - 10*ag[gen1[0], gen1[1]] + 100*takecity
                            + 2*genliberties)

                notgen1moves = [move for move in moves if move[0] != gen1[0]
                               or move[1] != gen1[1]]
                move = max(notgen1moves, key=f) if notgen1moves else None

        # mid game (turns >25)
        else:
            seen_enemy = p2.sum() > 0

            # get nearest opponent that is next to fog
            if seen_enemy:
                fog = (tg == FOG) | (tg == OBSTACLE)
                enemyedge = adj(fog) & p2
                if enemyedge.sum():
                    targets = minwhere(distgen1, enemyedge)
                else:
                    targets = minwhere(distgen1, p2)
                # targets = maxwhere(distgen1, p2 & (distgen1 < INF))
                targetdist = bfs(targets[:1], state)
                path = get_path(gen1[0], gen1[1], targetdist)

            # if enemy not seen, gather at general
            else:
                path = np.zeros(p1.shape).astype(bool)
                path[gen1[0], gen1[1]] = True

            # path from general to target
            distpath = bfs(where(path), state)

            turn3 = int(turn / 25) % 3
            if turn > 250 and turn3 == 0:
                time_attack = False
            elif turn > 250 and turn3 == 2:
                time_attack = True
            else:
                time_attack = turn % 25 >= 25 - 2 - (len(path)+1)/2

            # attack
            if seen_enemy and time_attack:
                print "attack"

                def f(move):
                    s = helpers.apply_move(state, move)
                    ag = np.asarray(s['army_grid'])
                    tg = np.asarray(s['tile_grid'])
                    p1 = tg == pi
                    p2 = (tg >= 0) & (tg != pi)

                    attack = (ag * p1 * targetdist ** 1.1).sum() / (
                        ag * p1).sum()

                    merge = ag[p1].std()
                    darmy = 2 * s['armies'][pi] - sum(s['armies'])
                    capture = s['lands'][pi] - sum(s['lands'])

                    staypath = (ag * p1 * path).sum() / (ag * p1).sum()

                    # takecity = -(city * (~p1) * (tg >= 0)).sum()
                    takecity = -(city * p2).sum()
                    if (turn > 50 and takecities and safecity.sum()
                            and numcities1 < numcities2
                            and not isthreat):
                        takeemptycity = -1000*(safecity * (tg != pi)).sum()
                        # print move, takeemptycity, 3*(ag * p1 * distsafecity ** 1.1).sum() / (
                        #     ag * p1).sum()
                        takeemptycity -= 3*(
                            (ag * p1 * distsafecity ** 1.1).sum() / (
                            ag * p1).sum())
                    else:
                        takeemptycity = 0

                    # explore or go for enemy generals
                    if gens2:
                        a = -2 * (ag * p1 * distgens2 ** 1.1).sum() / (
                            ag * p1).sum()
                        canwin = ((tg * p1 * (distgens2 == 1)).sum() >
                                  (tg * p2 * (distgens2 == 0)).sum() + 2)
                    else:
                        oldp2 = (state['tile_grid'] >= 0) & (
                            state['tile_grid'] != pi)
                        a = 2.0*((~explored) & adj(p1 & oldp2)).sum()
                        a -= 0.03*(ag * p1 * distfarexplored ** 0.9).sum() / (
                            ag * p1).sum()
                        canwin = 0.0

                    if (isthreat and (ag[ty, tx] >
                            (ag+1)[threatpath & ochoke].sum()
                            or len(threatpath) <= 2)):
                        m = ochoke & (threat > 0)
                        defend = (-1000000 * p2[m].sum()
                            - (ag * p1 * distnearthreat * close2).sum()/(
                            ag * p1 * close2).sum()
                            + 1000 * (ag * (1.0*p1-1.0*p2) * close).sum())
                        # print move, -1000000 * p2[m].sum(), - (ag * p1 * distnearthreat * close2).sum()/(ag * p1 * close2).sum(), 1000 * (ag * (1.0*p1-1.0*p2) * close).sum()
                    else:
                        defend = 10**8 - 10*p2[distgen1 <= 2].sum()

                    return (-1000000*sum(s['alives']) + 500000*canwin
                            + 10*darmy + 2*capture
                            - attack + a + 100*takecity + takeemptycity
                            + 0.01*merge + 0.02*staypath + defend)# + 0.01*(p1*ag).max())

                move = max(moves + [None], key=f)

            # expand
            # elif (seen_enemy and turn % 25 < 6) or (
            #         not seen_enemy and turn < 50) or (
            #         not seen_enemy and turn % 25 > 17) or (
            #         not seen_enemy and turn % 25 < 10):
            elif (seen_enemy and turn < 50 and turn % 25 < 6) or (
                    seen_enemy and turn < 100 and turn % 25 < 4) or (
                    seen_enemy and turn < 150 and turn % 25 < 2) or (
                    seen_enemy and turn % 25 < 1) or (
                    not seen_enemy):
                print "expand"

                # grab land or move towards nearest empty to general
                def f(move):
                    s = helpers.apply_move(state, move)
                    ag = np.asarray(s['army_grid'])
                    tg = np.asarray(s['tile_grid'])
                    p1 = tg == pi
                    p2 = (tg >= 0) & (tg != pi)

                    darmy = 2 * s['armies'][pi] - sum(s['armies'])
                    dland = 2 * s['lands'][pi] - sum(s['lands'])
                    explore = (distfarp1 * p1 * ag).sum() / (p1 * ag).sum()

                    if gens2:
                        canwin = ((tg * p1 * (distgens2 == 1)).sum() >
                                  (tg * p2 * (distgens2 == 0)).sum() + 2)
                    else:
                        canwin = 0.0

                    if (isthreat and (ag[ty, tx] >
                            (ag+1)[threatpath & ochoke].sum()
                            or len(threatpath) <= 2)):
                        m = ochoke & (threat > 0)
                        defend = (-1000000 * p2[m].sum()
                            - (ag * p1 * distnearthreat * close2).sum()/(
                            ag * p1 * close2).sum()
                            + 1000 * (ag * (1.0*p1-1.0*p2) * close).sum())
                        # print move, -1000000 * p2[m].sum(), - (ag * p1 * distnearthreat * close2).sum()/(ag * p1 * close2).sum(), 1000 * (ag * (1.0*p1-1.0*p2) * close).sum()
                    else:
                        defend = 10**8 - 10*p2[distgen1 <= 2].sum()
                    takecity = -(city * (~p1) * (tg >= 0)).sum()
                    # takecity = -(city * p2).sum()

                    # print move, 10*darmy, dland, 100*takecity, -explore, defend

                    return (-1000000*sum(s['alives']) + 500000*canwin
                            + 10*darmy + dland
                            + 100*takecity - explore + defend)

                move = max(moves + [None], key=f)

            # gather
            else:
                print "gather"

                # gather armies to path
                def f(move):
                    s = helpers.apply_move(state, move)
                    ag = np.asarray(s['army_grid'])
                    tg = np.asarray(s['tile_grid'])
                    p1 = tg == pi
                    p2 = (tg >= 0) & (tg != pi)

                    # darmy = 2 * s['armies'][pi] - sum(s['armies'])
                    c = (1.0 + 4*city)
                    groupup = (p1 * ag * distpath ** 1.1 / c).sum() / (
                        p1 * ag / c).sum()
                    merge = ag[p1].std()

                    if gens2:
                        canwin = ((tg * p1 * (distgens2 == 1)).sum() >
                                  (tg * p2 * (distgens2 == 0)).sum() + 2)
                    else:
                        canwin = 0.0

                    if (isthreat and (ag[ty, tx] >
                            (ag+1)[threatpath & ochoke].sum()
                            or len(threatpath) <= 2)):
                        m = ochoke & (threat > 0)
                        defend = (-1000000 * p2[m].sum()
                            - (ag * p1 * distnearthreat * close2).sum()/(
                            ag * p1 * close2).sum()
                            + 1000 * (ag * (1.0*p1-1.0*p2) * close).sum())
                        # print move, -1000000 * p2[m].sum(), - (ag * p1 * distnearthreat * close2).sum()/(ag * p1 * close2).sum(), 1000 * (ag * (1.0*p1-1.0*p2) * close).sum()
                    else:
                        defend = 10**8 - 10*p2[distgen1 <= 2].sum()

                    return (-1000000*sum(s['alives']) + 500000*canwin
                            + 10*s['armies'][pi]
                            + 0.01*merge - groupup + defend)

                move = max(moves + [None], key=f)

        if move:
            client.move(*move)


def minwhere(x, m):
    return where((x == x[m].min()) & m)


def maxwhere(x, m):
    return where((x == x[m].max()) & m)


def where(x):
    return zip(*np.where(x))


def bfs(initial, state, filt=None):
    pi = state['player_index']
    tg = state['tile_grid']
    dist = np.full((state['rows'], state['cols']), INF)
    for y, x in initial:
        dist[y, x] = 0

    q = collections.deque()
    q.extendleft(initial)
    seen = set(initial)
    while q:
        y, x = q.pop()
        for dy, dx in DIRECTIONS:
            by, bx = y+dy, x+dx
            if (0 <= by < state['rows'] and 0 <= bx < state['cols']
                    and (by, bx) not in seen
                    and (filt is None or filt[by, bx])):
                seen.add((by, bx))
                if (state['tile_grid'][by][bx] not in [
                            MOUNTAIN, OBSTACLE]):
                    dist[by, bx] = dist[y, x] + 1
                    if ((by, bx) not in state['cities']
                            or state['tile_grid'][by][bx] != EMPTY):
                        q.appendleft((by, bx))

    return dist


def get_path(y, x, dist):
    path = np.zeros(dist.shape).astype(bool)
    path[y, x] = True
    while dist[y, x]:
        for dy, dx in DIRECTIONS:
            by, bx = y+dy, x+dx
            if (0 <= by < dist.shape[0] and 0 <= bx < dist.shape[1]
                    and dist[by, bx] == dist[y, x] - 1):
                y, x = by, bx
                path[y, x] = True
                break
    return path


ADJ_FILTER = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
])


def adj(x):
    conv = convolve2d(x, ADJ_FILTER, mode='same')
    return (conv > 0) & (~x)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="na")
    parser.add_argument("--userid", required=True)
    parser.add_argument("--username")
    parser.add_argument("--mode", default='1v1')
    parser.add_argument("--gameid")
    args = parser.parse_args()
    start_game(args.region, args.userid, args.username, args.mode, args.gameid)
