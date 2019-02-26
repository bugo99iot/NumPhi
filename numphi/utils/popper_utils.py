import math
import numpy as np
import random

from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColorBar, ColorMapper, Ticker
from bokeh.models.sources import ColumnDataSource
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256 as palette  # this import will will be highlighted by PyCharm, ignore it
from numpy.core._multiarray_umath import ndarray

from numphi.parameters import INFLUENCE_OPTIONS, REINFORCE_OPTIONS
from typing import List
import logging
import os

# define logging level
logger = logging.getLogger(__name__)

if os.getenv('ENV') in ['staging', 'production']:
    logging.basicConfig(level=logging.WARNING)
else:
    logging.basicConfig(level=logging.INFO)


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


def get_all_neighbours_for_cell(coords: tuple) -> List[tuple]:
    # if y coordinate is even

    if coords[1] % 2 == 0:

        variations = [(0, -1), (-1, 0), (0, 1), (1, 0), (1, -1), (-1, -1)]

        cell_friends = list()

        for v in variations:
            new_coord = coords[0] + v[0], coords[1] + v[1]

            cell_friends.append(new_coord)

        # we need some extra juggling here to make sure that cells near the edge have as many friends as possible

        cell_friends_negative = [k for k in cell_friends if k[0] < 0 or k[1] < 0]
        cell_friends_positive = [k for k in cell_friends if k not in cell_friends_negative]

        random.shuffle(cell_friends_negative)
        random.shuffle(cell_friends_positive)

        yield cell_friends_positive + cell_friends_negative

    # if y coordinate is odd

    else:

        variations = [(0, -1), (-1, 0), (0, 1), (1, 0), (-1, 1), (1, 1)]

        cell_friends = list()

        for v in variations:
            new_coord = coords[0] + v[0], coords[1] + v[1]

            cell_friends.append(new_coord)

        # we need some extra juggling here to make sure that cells near the edge have as many friends as possible

        cell_friends_negative = [k for k in cell_friends if k[0] < 0 or k[1] < 0]
        cell_friends_positive = [k for k in cell_friends if k not in cell_friends_negative]

        random.shuffle(cell_friends_negative)
        random.shuffle(cell_friends_positive)

        yield cell_friends_positive + cell_friends_negative


def get_coordinates_of_all_cells(board_side: int) -> list:

    if board_side < 2:

        raise Exception("Board side must be 2 or greater")

    all_coordinates = list()

    for v in range(board_side):

        for h in range(board_side):

            all_coordinates.append((v, h))

    return all_coordinates


def generate_n_friends(center: tuple, board_side: int) -> List[tuple]:

    friends = list()

    starting_cells = [center]

    for step in range(0, board_side - 1):

        step_neighbours = list()

        for cell in starting_cells:

            cell_neighbours = get_all_neighbours_for_cell(coords=cell)

            for neighbour in cell_neighbours:

                if neighbour not in friends and neighbour != center:

                    step_neighbours.append(neighbour)

                    friends.append(neighbour)

                    yield neighbour

            starting_cells = step_neighbours


def build_interaction_matrix(friend_cells: int, share_active: float, board_side: int) -> np.ndarray:

    if friend_cells > int(board_side**2):

        logging.warning("Friend cells > board size, friend cells will be board size -1")

        friend_cells = int(board_side**2) - 1

    influence_matrix = np.array([] for _ in range(board_side * board_side)).reshape(board_side, board_side)

    for coords, _ in np.ndenumerate(influence_matrix):

        current_friends = list()

        for i in range(friend_cells):

            new_cell = generate_n_friends(center=coords, board_side=board_side)

            if new_cell[0] >= 0 and new_cell[1] >= 0:

                current_friends.append(new_cell)

        influence_matrix[coords] = current_friends

    return influence_matrix


def offset_to_axial(coords: tuple) -> tuple:
    """
    Convert odd-q offset coordinates to axial coordinates (needed by Bokeh)
    see https://www.redblobgames.com/grids/hexagons/

    :param coords: e.g. (0, 1)
    :return:
    """

    return coords[0] - math.floor(coords[1] / 2.0), coords[1]


def plot_hextile(checkboard: np.ndarray):

    data = [(coords, cell.t, cell.a, cell.d) for coords, cell in np.ndenumerate(checkboard)]

    coords_offset, t, a, d = list(zip(*data))

    coords_axial = [offset_to_axial(c) for c in coords_offset]

    r_axial, q_axial = list(zip(*coords_axial))

    r_axial: ndarray = np.array(r_axial)
    q_axial: ndarray = np.array(q_axial)
    t: ndarray = np.array(t)

    size = 0.5
    orientation = "flattop"

    t_tolerant = t[t > 0.5]

    if (t.size - t_tolerant.size) / t.size >= 0.5:

        title = "Intolerant"

    else:
        title = "Tolerant"

    p = figure(title=title, match_aspect=True, tools="wheel_zoom,reset", plot_width=500, plot_height=500)

    p.title.align = 'center'

    source = ColumnDataSource(data=dict(
        q=q_axial,
        r=r_axial,
        c=t,
        a=a,
        d=d))

    p.hex_tile(q="q", r="r", size=size,
               fill_color=linear_cmap('c', palette, 0.0, 1.0),
               line_color=None, source=source, orientation=orientation)

    p.xaxis.visible = False
    p.yaxis.visible = False

    #p.background_fill_color = '#440154'
    p.grid.visible = False

    hover = HoverTool(tooltips=[("tolerance", "@c"), ("attack", "@a"), ("defence", "@d")])

    p.add_tools(hover)


    show(p)
