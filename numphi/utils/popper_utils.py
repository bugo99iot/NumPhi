import math
import numpy as np
import matplotlib.pyplot as plt
import random


def is_square(integer: int):
    """
    Check if an int is a square number

    :param integer:
    :return:
    """

    root = math.sqrt(integer)

    if int(root + 0.5) ** 2 == integer:

        return True

    else:

        return False


def print_checkboard(checkboard: np.ndarray, colors: tuple) -> None:
    """

    :param checkboard:
    :param colors:
    :return:
    """

    # todo: show in plot epoch and abs number of tol / intol defined with 0.5 threshold

    side = checkboard.shape[0]

    checkboard_linear = checkboard.reshape(-1)

    checkboard_linear = np.array([k.t for k in checkboard_linear]).reshape(side, side)

    plt.matshow(checkboard_linear)

    plt.show()


def get_all_combos(coords: tuple, interaction_step: int, board_side: int) -> list:
    """

    :param coords:
    :param interaction_step:
    :param board_side:
    :return:
    """

    if coords[0] >= board_side or coords[1] >= board_side:

        raise Exception("Coords leger than board side for coords: {}".format(coords))

    hr = np.arange(coords[0] - interaction_step, coords[0] + interaction_step + 1)

    vr = np.arange(coords[1] - interaction_step, coords[1] + interaction_step + 1)

    combos = list()

    for h in hr:

        if h < 0 or h >= board_side:

            continue

        for v in vr:

            if v < 0 or v >= board_side:
                continue

            if (h, v) == coords:

                continue

            combos.append((h, v))

    random.shuffle(combos)

    return combos


def get_influenced_t_after_influence(influenced_t: float, influencer_t: float,
                                     influenced_a: float, influencer_a: float,
                                     influenced_d: float, influencer_d: float,
                                     direction: str = "lower") -> float:

    min_influence = 0.01

    # if influencer is not enough influential, influenced_t is preserved

    if influencer_a <= influenced_d:

        return influenced_t

    # if influenced is enough influential, influenced_t will be modified

    amount = round((influencer_t - influenced_t) / 10.0, 2)

    if amount < 0.0:

        if direction == "bi":

            amount = abs(amount)

            if amount < min_influence:

                amount = min_influence

            influenced_t += amount

    elif amount > 0.0:

        if amount < min_influence:

            amount = min_influence

        influenced_t -= amount

    if influenced_t > 1.0:

        influenced_t = 1.0

    if influenced_t < 0.0:

        influenced_t = 0.0

    return influenced_t
