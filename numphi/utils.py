import math
import numpy as np
import matplotlib.pyplot as plt


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


def print_checkboard(actors_list: list, colors: tuple) -> None:
    """
    Show checkboard

    :param actors_list:
    :return:
    """

    actors_to_print = np.asarray([k.t for k in actors_list])

    board_side = int(math.sqrt(len(actors_list)))

    image = actors_to_print.reshape((board_side, board_side))

    plt.matshow(image)

    plt.show()
