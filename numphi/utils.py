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
    Show checkboard

    :param checkboard:
    :return:
    """

    side = checkboard.shape[0]

    checkboard_linear = checkboard.reshape(-1)

    checkboard_linear = np.array([k.t for k in checkboard_linear]).reshape(side, side)

    plt.matshow(checkboard_linear)

    plt.show()


def get_all_combos(coords: tuple, range: int, board_side: int) -> list:
    """

    :param hr:
    :param vr:
    :param board_side:
    :return:
    """

    if coords[0] >= board_side or coords[1] >= board_side:

        raise Exception("Coords leger than board side for coords: {}".format(coords))

    hr = np.arange(coords[0] - range, coords[0] + range + 1)

    vr = np.arange(coords[1] - range, coords[1] + range + 1)

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

    random. shuffle(combos)

    return combos
