
import copy
import generals
import numpy as np


DIRECTIONS = [
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
]
OBSTACLES = [generals.MOUNTAIN, generals.OBSTACLE]


def get_all_moves(state):
    moves = []
    for y in range(state['rows']):
        for x in range(state['cols']):
            moves += get_moves(state, y, x)
    return moves


def get_moves(state, y, x):
    if (state['tile_grid'][y][x] != state['player_index']
            or state['army_grid'][y][x] < 2):
        return []

    moves = []
    for dy, dx in DIRECTIONS:
        by, bx = y + dy, x + dx
        if (0 <= by < state['rows']
                and 0 <= bx < state['cols']
                and state['tile_grid'][by][bx] not in OBSTACLES):
            moves.append((y, x, by, bx))
    return moves


def apply_move(state, move):
    state = copy.deepcopy(state)
    pi = state['player_index']
    ag = state['army_grid'] = np.asarray(state['army_grid'])
    tg = state['tile_grid'] = np.asarray(state['tile_grid'])
    if move is not None:
        ay, ax, by, bx = move
        assert abs(ay - by) + abs(ax - bx) == 1

        if (tg[ay, ax] == pi
                and tg[by, bx] not in OBSTACLES
                and ag[ay, ax] > 1):
            n = ag[ay, ax] - 1

            # neutral city or enemy tile
            if (((by, bx) in state['cities'] and tg[by, bx] == generals.EMPTY)
                    or (tg[by, bx] >= 0 and tg[by, bx] != pi)):
                if n > ag[by, bx]:
                    m = ag[by, bx]
                    state['armies'][pi] -= m
                    state['lands'][pi] += 1
                    if tg[by, bx] >= 0:
                        state['armies'][tg[by, bx]] -= m
                        state['lands'][tg[by, bx]] -= 1
                    ag[ay, ax] -= n
                    ag[by, bx] = n - m
                    tg[by, bx] = pi
                else:
                    state['armies'][pi] -= n
                    if tg[by, bx] >= 0:
                        state['armies'][tg[by, bx]] -= n
                    ag[ay, ax] -= n
                    ag[by, bx] -= n

            # friendly tile
            elif tg[by, bx] == pi:
                ag[ay, ax] -= n
                ag[by, bx] += n

            # empty tile
            elif tg[by, bx] == generals.EMPTY:
                state['lands'][pi] += 1
                ag[ay, ax] -= n
                ag[by, bx] += n
                tg[by, bx] = pi

            else:
                raise AssertionError()

    # generate armies
    state['turn'] += 1
    if state['turn'] % 25 == 0:
        for p in np.unique(tg):
            if p >= 0:
                state['armies'][p] += (tg == p).sum()
        ag[tg >= 0] += 1
    else:
        for y, x in state['generals']:
            if y >= 0 and x >= 0:
                state['armies'][tg[y, x]] += 1
                ag[y, x] += 1
        for y, x in state['cities']:
            if tg[y, x] >= 0:
                state['armies'][tg[y, x]] += 1
                ag[y, x] += 1

    for i, (y, x) in enumerate(state['generals']):
        if y >= 0 and x >= 0 and tg[y, x] != i:
            state['alives'][i] = False

    return state
