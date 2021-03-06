import math
import numpy as np
import random

from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, Slider, CustomJS
from bokeh.models.sources import ColumnDataSource
from bokeh.transform import linear_cmap
from bokeh.layouts import column, widgetbox
from bokeh.palettes import Plasma256 as palette  # this import will will be highlighted by PyCharm, ignore it
from bokeh.models.widgets import Panel, Tabs

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
        return upper_bound

    if value < lower_bound:
        return lower_bound

    return value


def get_influenced_t_after_influence(influenced_t: float, influencer_t: float,
                                     influenced_a: float, influencer_a: float,
                                     influenced_d: float, influencer_d: float,
                                     direction: str) -> float:
    if direction is None or isinstance(direction, str) is False or direction not in INFLUENCE_OPTIONS:
        raise Exception("influence must be a string among: {}".format(INFLUENCE_OPTIONS))

    if influenced_t == influencer_t:

        return influenced_t

    elif influenced_t > influencer_t:

        return round(bound_value(influenced_t - 0.01), 2)

    else:

        return influenced_t


def get_influenced_a_after_influence(influenced_t: float, influencer_t: float,
                                     influenced_a: float, influencer_a: float,
                                     influenced_d: float, influencer_d: float,
                                     direction: str) -> float:

    if direction is None or isinstance(direction, str) is False or direction not in REINFORCE_OPTIONS:
        raise Exception("influence must be a string among: {}".format(REINFORCE_OPTIONS))

    change_in_attack = 0.01

    influencer_a - influenced_a

    if influencer_a > influenced_a <= 0:
        return influenced_a - change_in_attack

    return influenced_a


def get_influenced_d_after_influence(influenced_t: float, influencer_t: float,
                                     influenced_a: float, influencer_a: float,
                                     influenced_d: float, influencer_d: float,
                                     direction: str) -> float:
    if direction is None or isinstance(direction, str) is False or direction not in REINFORCE_OPTIONS:
        raise Exception("influence must be a string among: {}".format(REINFORCE_OPTIONS))

    change_in_defense = 0.01

    tolerance_gap = influencer_t - influenced_t

    if tolerance_gap <= 0:

        return influenced_d - change_in_defense

    return influenced_d


def get_all_neighbours_for_cell(coords: tuple, board_side: int) -> List[tuple]:

    # if y coordinate is even

    if coords[1] % 2 == 0:

        variations = [(0, -1), (-1, 0), (0, 1), (1, 0), (-1, 1), (-1, -1)]

        cell_friends = list()

        for v in variations:
            new_coord = coords[0] + v[0], coords[1] + v[1]

            cell_friends.append(new_coord)

        # we need some extra juggling here to make sure that cells near the edge have as many friends as possible

        cell_friends_inside_board = [k for k in cell_friends if 0 <= k[0] < board_side and 0 <= k[1] < board_side]

        # random.shuffle(cell_friends_inside_board)

        return cell_friends_inside_board

    # if y coordinate is odd

    else:

        variations = [(0, -1), (-1, 0), (0, 1), (1, 0), (1, -1), (1, 1)]

        cell_friends = list()

        for v in variations:
            new_coord = coords[0] + v[0], coords[1] + v[1]

            cell_friends.append(new_coord)

        # we need some extra juggling here to make sure that cells near the edge have as many friends as possible

        cell_friends_inside_board = [k for k in cell_friends if 0 <= k[0] < board_side and 0 <= k[1] < board_side]

        # random.shuffle(cell_friends_inside_board)

        return cell_friends_inside_board


def generate_n_friends(center: tuple, board_side: int):

    friends = list()

    starting_cells = [center]

    for step in range(0, board_side):

        step_neighbours = list()

        for cell in starting_cells:

            for neighbour in get_all_neighbours_for_cell(coords=cell, board_side=board_side):

                if neighbour not in friends and neighbour != center:

                    step_neighbours.append(neighbour)

                    friends.append(neighbour)

                    yield neighbour

            starting_cells = step_neighbours


def build_interaction_matrix(friend_cells: int, share_active: float, board_side: int) -> np.ndarray:

    if friend_cells > int(board_side ** 2):
        logging.warning("Friend cells > board size, friend cells will be board size -1")

        friend_cells = int(board_side ** 2) - 1

    influence_matrix = np.empty([board_side, board_side], dtype=object)

    for coords, _ in np.ndenumerate(influence_matrix):

        current_friends = list()

        for new_cell in generate_n_friends(center=coords, board_side=board_side):

            current_friends.append(new_cell)

            if len(current_friends) == friend_cells:
                break

        influence_matrix[coords] = current_friends

    return influence_matrix


def offset_to_axial(coords: tuple) -> tuple:
    """
    Convert odd-r offset coordinates to axial coordinates (needed by Bokeh)
    see https://www.redblobgames.com/grids/hexagons/

    :param coords: e.g. (0, 1)
    :return:
    """

    return coords[1], coords[0] - math.floor(coords[1] / 2.0)


def plot_hextile(checkboard: np.ndarray):

    data = [(coords, cell.t, cell.a, cell.d) for coords, cell in np.ndenumerate(checkboard)]

    coords_offset, t, a, d = list(zip(*data))

    r_offset, q_offset = list(zip(*coords_offset))

    coords_axial = [offset_to_axial(c) for c in coords_offset]

    r_axial, q_axial = list(zip(*coords_axial))

    r_axial: np.ndarray = np.array(r_axial)
    q_axial: np.ndarray = np.array(q_axial)
    t: np.ndarray = np.array(t)

    size = 0.5
    orientation = "pointytop"

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
        d=d,
        r_offset=r_offset,
        q_offset=q_offset))

    p.hex_tile(q="q", r="r", size=size,
               fill_color=linear_cmap('c', palette, 0.0, 1.0),
               line_color=None, source=source, orientation=orientation,
               hover_color="pink", hover_alpha=0.8)

    p.xaxis.visible = False
    p.yaxis.visible = False

    # p.background_fill_color = '#440154'
    p.grid.visible = False

    path_one_up = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1]) + "/sample_data"

    output_file(path_one_up + "/demo_hextile.html")

    hover = HoverTool(tooltips=[("tolerance", "@c{0.00}"), ("attack", "@a{0.00}"), ("defence", "@d{0.00}"),
                                ("r_offset, q_offset", "@r_offset, @q_offset")])

    p.add_tools(hover)

    show(p)


# todo: add function to plot cell with given coordinates and n friends in m sided checkboard
def plot_cell_friends_hextile(center: tuple, board_side):
    n_cells = int(board_side ** 2)

    checkboard = np.array([0.0 for _ in range(n_cells)]).reshape

    # color 0.0 target, 0.5 friends, 1.0 others

    return None


def get_data_dict(checkboard: np.ndarray, epoch: int):

    full_dict = dict()

    data = [(coords, cell.t, cell.a, cell.d) for coords, cell
            in np.ndenumerate(checkboard)]

    coords_offset, t, a, d = list(zip(*data))

    r_offset, q_offset = list(zip(*coords_offset))

    coords_axial = [offset_to_axial(c) for c in coords_offset]

    r_axial, q_axial = list(zip(*coords_axial))

    r_axial = np.array(r_axial)
    q_axial = np.array(q_axial)
    t = np.array(t)

    epoch_array = np.full(t.size, epoch)

    full_dict["q_axial"] = q_axial
    full_dict["r_axial"] = r_axial
    full_dict["t"] = t
    full_dict["a"] = a
    full_dict["d"] = d
    full_dict["r_offset"] = r_offset
    full_dict["q_offset"] = q_offset
    full_dict["epoch"] = epoch_array  # tweak to fix frontend issue

    return full_dict


def plot_bokeh_board(iterable_checkboards: List[np.ndarray]):

    path_one_up = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1]) + "/sample_data"

    output_file(path_one_up + "/demo_hextile.html")

    sources = [ColumnDataSource(get_data_dict(checkboard=checkboard, epoch=i)) for i, checkboard
               in enumerate(iterable_checkboards)]

    sources = sources + [ColumnDataSource(get_data_dict(checkboard=iterable_checkboards[0], epoch=0))]

    source = sources[-1]

    size = 0.5
    orientation = "pointytop"

    t = source.data.get('t')

    t_tolerant = t[t > 0.5]

    if (t.size - t_tolerant.size) / t.size >= 0.5:

        title = "Intolerant"

    else:
        title = "Tolerant"

    p = figure(title=title, match_aspect=True, tools="wheel_zoom,reset", plot_width=500, plot_height=500)

    p.title.align = 'center'

    p.hex_tile(q="q_axial", r="r_axial", size=size,
               fill_color=linear_cmap('t', palette, 0.0, 1.0),
               line_color=None, source=source, orientation=orientation,
               hover_color="pink", hover_alpha=0.8)

    p.xaxis.visible = False
    p.yaxis.visible = False

    # p.background_fill_color = '#440154'
    p.grid.visible = False

    # Add the slider
    code = """       
        var step = cb_obj.value;    
        var new_data = sources[step].data;
        source.data = new_data;
        source.change.emit();   
        """

    callback = CustomJS(args=dict(source=source, sources=sources), code=code)

    slider = Slider(start=0, end=len(sources)-2, step=1, value=0, title="Epoch", callback=callback)

    hover = HoverTool(tooltips=[("tolerance", "@t{0.00}"),
                                ("attack", "@a{0.00}"),
                                ("defence", "@d{0.00}"),
                                ("r_offset, q_offset", "@r_offset, @q_offset"),
                                ("r_axial, q_axial", "@r_axial, @q_axial"),
                                ("epoch", "@epoch")], callback=callback)

    p.add_tools(hover)

    # annotations: http://bokeh.pydata.org/en/latest/docs/user_guide/annotations.html#userguide-annotations

    layout = column(p, widgetbox(slider))

    tab1 = Panel(child=layout, title="Simulation")

    p2 = figure(title="Coming soon", match_aspect=True, tools="wheel_zoom,reset", plot_width=500, plot_height=500)
    p2.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=3, color="navy", alpha=0.5)
    tab2 = Panel(child=p2, title="Coming soon")

    tabs = Tabs(tabs=[tab1, tab2])

    show(tabs)
