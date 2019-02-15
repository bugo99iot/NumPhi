import math
import numpy as np
import matplotlib.pyplot as plt
import random

from numphi.parameters import INFLUENCE_OPTIONS, REINFORCE_OPTIONS


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


def print_checkboard(checkboard: np.ndarray, cmap: str, epoch: int) -> None:
    """

    :param checkboard:
    :param colors:
    :return:
    """

    # todo: show in plot epoch and abs number of tol / intol defined with 0.5 threshold

    side = checkboard.shape[0]

    checkboard_linear = checkboard.reshape(-1)

    checkboard_tolerance = np.array([k.t for k in checkboard_linear]).reshape(side, side)

    checkboard_attack = np.array([k.a for k in checkboard_linear]).reshape(side, side)

    checkboard_defense = np.array([k.d for k in checkboard_linear]).reshape(side, side)

    plt.figure(1)

    plt.subplot(121)
    plt.imshow(checkboard_tolerance, vmin=0.0, vmax=1.0, cmap=cmap)
    plt.gca().set_title("Tolerance")

    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)

    plt.imshow(checkboard_attack, vmin=0.0, vmax=1.0, cmap=cmap)

    plt.gca().set_title("Attack")

    plt.xticks([])
    plt.yticks([])

    plt.suptitle("Epoch: {}".format(epoch))

    plt.show()

    """    plt.imshow(checkboard_tolerance, vmin=0.0, vmax=1.0, cmap=cmap)

    plt.suptitle("Epoch: {}".format(epoch))

    plt.show()"""


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


def bound_value(value: float):
    """
    Return value within bounds

    :param value:
    :return:
    """

    upper_bound = 1.0
    lower_bound = 0.0

    if value > upper_bound:

        return 1.0

    if value < lower_bound:

        return 0.0

    return value


def get_influenced_t_after_influence(influenced_t: float, influencer_t: float,
                                     influenced_a: float, influencer_a: float,
                                     influenced_d: float, influencer_d: float,
                                     direction: str) -> float:

    if direction is None or isinstance(direction, str) is False or direction not in INFLUENCE_OPTIONS:
        raise Exception("influence must be a string among: {}".format(INFLUENCE_OPTIONS))

    if direction == "none":

        return influenced_t

    min_influence = 0.01

    # if influencer is not enough influential, influenced_t is preserved

    if influencer_a <= influenced_d:

        return influenced_t

    # if influenced is enough influential, influenced_t will be modified

    tolerance_gap = influenced_t - influencer_t

    # if influencer is more tolerant and direction is drag up, then influenced becomes more tolerant

    if tolerance_gap < 0.0:

        if direction in ["drag_up", "always"]:

            amount = abs(round(tolerance_gap / 10.0, 2))

            if amount < min_influence:

                amount = min_influence

            influenced_t += amount

        return bound_value(value=influenced_t)

    # if influencer is more intolerant, then influenced becomes more intolerant

    elif tolerance_gap > 0.0:

        if direction in ["drag_down", "always"]:

            amount = round(tolerance_gap / 10.0, 2)

            if amount < min_influence:

                amount = min_influence

            influenced_t -= amount

            return bound_value(value=influenced_t)

    return bound_value(value=influenced_t)


def get_influenced_a_after_influence(influenced_t: float, influencer_t: float,
                                     influenced_a: float, influencer_a: float,
                                     influenced_d: float, influencer_d: float,
                                     direction: str) -> float:

    if direction is None or isinstance(direction, str) is False or direction not in REINFORCE_OPTIONS:
        raise Exception("influence must be a string among: {}".format(REINFORCE_OPTIONS))

    if direction == "none":
        return influenced_a

    tolerance_gap_for_similarity = 0.2

    change_in_attack = 0.01

    tolerance_gap = influencer_t - influenced_t

    max_tolerance = max(influenced_t, influencer_t)

    min_tolerance = min(influenced_t, influencer_t)

    # case in which both cells are tolerant

    if min_tolerance > 0.5:

        if direction in ["always", "when_tolerant"]:

            if abs(tolerance_gap) <= tolerance_gap_for_similarity:

                influenced_a += change_in_attack

                influenced_a = round(influenced_a, 2)

                return bound_value(value=influenced_a)

            elif abs(tolerance_gap) > tolerance_gap_for_similarity:

                influenced_a -= change_in_attack

                influenced_a = round(influenced_a, 2)

                return bound_value(value=influenced_a)

    # case in which both cells are intolerant

    elif max_tolerance < 0.5:

        if direction in ["always", "when_intolerant"]:

            if abs(tolerance_gap) <= tolerance_gap_for_similarity:

                influenced_a += change_in_attack

                influenced_a = round(influenced_a, 2)

                return bound_value(value=influenced_a)

            elif abs(tolerance_gap) > tolerance_gap_for_similarity:

                influenced_a -= change_in_attack

                influenced_a = round(influenced_a, 2)

                return bound_value(value=influenced_a)

    else:

        # if average is exactly 0.5 or one is tolerant, one intolerant, don't perform any action unless direction is
        # always

        if direction == "always":

            if abs(tolerance_gap) <= tolerance_gap_for_similarity:

                influenced_a += change_in_attack

                influenced_a = round(influenced_a, 2)

                return bound_value(value=influenced_a)

            elif abs(tolerance_gap) > tolerance_gap_for_similarity:

                influenced_a -= change_in_attack

                influenced_a = round(influenced_a, 2)

                return bound_value(value=influenced_a)

        return influenced_a

    return influenced_a


def get_influenced_d_after_influence(influenced_t: float, influencer_t: float,
                                     influenced_a: float, influencer_a: float,
                                     influenced_d: float, influencer_d: float,
                                     direction: str) -> float:

    if direction is None or isinstance(direction, str) is False or direction not in REINFORCE_OPTIONS:
        raise Exception("influence must be a string among: {}".format(REINFORCE_OPTIONS))

    if direction == "none":
        return influenced_d

    tolerance_gap_for_similarity = 0.2

    change_in_attack = 0.01

    tolerance_gap = influencer_t - influenced_t

    max_tolerance = max(influenced_t, influencer_t)

    min_tolerance = min(influenced_t, influencer_t)

    # case in which both cells are tolerant

    if min_tolerance > 0.5:

        if direction in ["always", "when_tolerant"]:

            if abs(tolerance_gap) <= tolerance_gap_for_similarity:

                influenced_d += change_in_attack

                influenced_d = round(influenced_d, 2)

                return bound_value(value=influenced_d)

            elif abs(tolerance_gap) > tolerance_gap_for_similarity:

                influenced_d -= change_in_attack

                influenced_d = round(influenced_d, 2)

                return bound_value(value=influenced_d)

    # case in which both cells are intolerant

    elif max_tolerance < 0.5:

        if direction in ["always", "when_intolerant"]:

            if abs(tolerance_gap) <= tolerance_gap_for_similarity:

                influenced_d += change_in_attack

                influenced_d = round(influenced_d, 2)

                return bound_value(value=influenced_d)

            elif abs(tolerance_gap) > tolerance_gap_for_similarity:

                influenced_d -= change_in_attack

                influenced_d = round(influenced_d, 2)

                return bound_value(value=influenced_d)

    else:

        # if average is exactly 0.5 or one is tolerant, one intolerant, don't perform any action unless direction is
        # always

        if direction == "always":

            if abs(tolerance_gap) <= tolerance_gap_for_similarity:

                influenced_d += change_in_attack

                influenced_d = round(influenced_d, 2)

                return bound_value(value=influenced_d)

            elif abs(tolerance_gap) > tolerance_gap_for_similarity:

                influenced_d -= change_in_attack

                influenced_d = round(influenced_d, 2)

                return bound_value(value=influenced_d)

        return influenced_d

    return influenced_d
