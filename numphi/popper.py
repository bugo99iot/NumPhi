import numpy as np
import logging
import math as math
import random
import copy

from numphi.parameters import COLORS_ALLOWED, INFLUENCE_OPTIONS, REINFORCE_OPTIONS
from numphi.exceptions import CheckBoardException, CellException
from numphi.utils.popper_utils import is_square, print_checkboard, get_all_combos_with_step, get_influenced_t_after_influence, \
    get_influenced_a_after_influence, get_influenced_d_after_influence

from dotenv import load_dotenv, find_dotenv
import os

# load env variables
load_dotenv(find_dotenv())

# define logging level
logger = logging.getLogger(__name__)

if os.getenv('ENV') in ['staging', 'production']:
    logging.basicConfig(level=logging.WARNING)
else:
    logging.basicConfig(level=logging.INFO)


# todo: add Sentry hook

# todo: add interaction step dump, interaction becomes weaker

# todo: add is bounded problem?

# todo check is reinforcement as effective for the tolerant as for the intolerant

# todo: add media (supercell)

# todo: add social media (can be in communication with far away )

# todo: show number of tol and intol >= 0.5 in plot

# todo: develop plot with slider

# todo add more start like half or bands or circles

class CheckBoard(object):

    def __init__(self, total_cells: int, friend_cells: int,
                 start: str = "random", start_proportion_intolerant: float = None, influence: str = "always",
                 reinforce: str = "always", share_active: float = 1.0, cmap: str = "spring"):

        if isinstance(total_cells, int) is False or total_cells < 4 or total_cells > 1000000:

            raise CheckBoardException("n_cells must be a square number integer n so that 9 <= n <= 1,000,000")

        if is_square(total_cells) is False:

            raise CheckBoardException("n_cells must be a square number")

        self.n_cells = total_cells

        if share_active is None or isinstance(share_active, float) is False \
                or (0.0 <= share_active <= 1.0) is False:

            raise CheckBoardException("share_active must be a float between 0 and 1")

        self.share_active = share_active

        self.board_side = int(math.sqrt(self.n_cells))

        if friend_cells is None or isinstance(friend_cells, int) is False or friend_cells < 1:

            raise CheckBoardException("friend_cells must be an integer > 0")

        self.friend_cells = friend_cells

        if self.friend_cells >= self.n_cells - 1:

            logging.warning("friend_cells is large, each cell will interacts with full board")

            self.friend_cells = self.n_cells - 1

        if reinforce is None or isinstance(reinforce, str) is False or reinforce not in REINFORCE_OPTIONS:

            raise CheckBoardException("reinforce must be a string among: {}".format(REINFORCE_OPTIONS))

        self.reinforce = reinforce

        if influence is None or isinstance(influence, str) is False or influence not in INFLUENCE_OPTIONS:

            raise CheckBoardException("influence must be a string among: {}".format(INFLUENCE_OPTIONS))

        self.influence = influence

        if cmap is None or isinstance(cmap, str) is False:

            raise CheckBoardException("cmap must be one of the following: https://matplotlib.org/examples/color/colormap"
                                      "s_reference.html")

        self.cmap = cmap

        # todo: check start

        self.start = start.lower() if start is not None else None

        self.start_proportion_intolerant = start_proportion_intolerant

        if self.start_proportion_intolerant is not None:

            if isinstance(self.start_proportion_intolerant, float) is False or (0.0 <= self.start_proportion_intolerant
                                                                                <= 1.0) is False:

                raise CheckBoardException("start_proportion must be a number between 0.0 and 1.0")

        if start == "random":

            self.checkboard = np.array([Cell(t=round(random.random(), 2),
                                             a=round(random.random(), 2),
                                             d=round(random.random(), 2)) for _ in
                                        range(self.n_cells)])

            np.random.shuffle(self.checkboard)

            self.checkboard = self.checkboard.reshape(self.board_side, self.board_side)

        elif start == "popper":

            if start_proportion_intolerant is None:

                raise CheckBoardException("If you choose Popper as start, you need to set start_proportion variable, "
                                          "e.g. start_proportion=0.9")

            n_intolerant = int(start_proportion_intolerant * self.n_cells)

            n_tolerant = self.n_cells - n_intolerant

            logging.info("Intolerant: {}, Tolerant: {}".format(n_intolerant, n_tolerant))

            intol = np.array([Cell(t=0.0, a=1.0, d=1.0) for _ in range(n_intolerant)])

            tol = np.array([Cell(t=1.0, a=0.0, d=0.0) for _ in range(n_tolerant)])

            intol_tol = np.concatenate((intol, tol), axis=0)

            np.random.shuffle(intol_tol)

            self.checkboard = intol_tol.reshape(self.board_side, self.board_side)

        else:

            raise CheckBoardException("Start not understood")

    def print_checkboard(self):
        """
        Print current checkboard

        :return:
        """

        print_checkboard(checkboard=self.checkboard, cmap=self.cmap, epoch="manual")

        return None

    def print_distribution(self):
        """
        Print tolerance distribution in 0.1 buckets
        :return:
        """

        return None

    def interact_n_times(self, n_of_interactions: int = 1) -> None:

        new_board = copy.copy(self.checkboard)

        interaction_matrix = build_interaction_matrix(friend_cells=self.friend_cells, share_active=self.share_active, board_side=self.board_side)

        for steps in range(n_of_interactions):

            step_board = copy.copy(new_board)

            for coords, cell in np.ndenumerate(self.checkboard):

                # find all cells being influenced by current cell

                all_combos = get_all_combos_with_step(coords=coords, interaction_step=self.interaction_step,
                                                      board_side=self.board_side)

                if self.share_active < 1.0:

                    # todo: make elegant
                    cut_index = int(round(random.uniform(self.share_active, 1.0) * len(all_combos)))

                    # each cell should interact at least with one cell
                    if cut_index == 0:

                        cut_index = 1

                    all_combos = all_combos[:cut_index]

                for combo in all_combos:

                    influenced = influence(influenced=new_board[combo], influencer=cell, direction=self.influence)

                    influenced = reinforce(influenced=influenced, influencer=cell, direction=self.reinforce)

                    step_board[combo] = influenced

            new_board = copy.copy(step_board)

            #print_checkboard(checkboard=new_board, cmap=self.cmap, epoch=steps+1)

        self.checkboard = new_board

        return self.checkboard


class Cell(object):

    def __init__(self, t: float, a: float, d: float, r: int = 1):
        """

        :param t: tolerance
        :param a: attack
        :param d: defense
        :param r: range
        """

        if isinstance(t, float) is False or (0.0 <= t <= 1.0) is False:

            raise CellException("t must be a float between 0.0 and 1.0")

        if isinstance(a, float) is False or (0.0 <= a <= 1.0) is False:

            raise CellException("a must be a float between 0.0 and 1.0")

        if isinstance(d, float) is False or (0.0 <= d <= 1.0) is False:

            raise CellException("d must be a float between 0.0 and 1.0")

        if isinstance(r, int) is False or r < 1:

            raise CellException("r must be int >= 1")

        self.t = t
        self.a = a
        self.d = d
        self.r = r


def influence(influenced: Cell, influencer: Cell, direction: str) -> Cell:

    if direction not in INFLUENCE_OPTIONS:
        raise Exception("direction must be among: {}".format(INFLUENCE_OPTIONS))

    # define brand new cell to avoid changing original
    cell_after_influence = copy.copy(influenced)

    # if influencer is enough influential, tolerance of influential will be modified

    cell_after_influence.t = get_influenced_t_after_influence(influenced_t=influenced.t, influencer_t=influencer.t,
                                                              influenced_a=influenced.a, influencer_a=influencer.a,
                                                              influenced_d=influenced.d, influencer_d=influencer.d,
                                                              direction=direction)

    return cell_after_influence


def reinforce(influenced: Cell, influencer: Cell, direction) -> Cell:

    if direction not in REINFORCE_OPTIONS:
        raise Exception("direction must be among: {}".format(REINFORCE_OPTIONS))

    # define brand new cell to avoid changing original
    cell_after_influence = copy.copy(influenced)

    cell_after_influence.a = get_influenced_a_after_influence(influenced_t=influenced.t, influencer_t=influencer.t,
                                                              influenced_a=influenced.a, influencer_a=influencer.a,
                                                              influenced_d=influenced.d, influencer_d=influencer.d,
                                                              direction=direction)

    cell_after_influence.d = get_influenced_d_after_influence(influenced_t=influenced.t, influencer_t=influencer.t,
                                                              influenced_a=influenced.a, influencer_a=influencer.a,
                                                              influenced_d=influenced.d, influencer_d=influencer.d,
                                                              direction=direction)

    return cell_after_influence


if __name__ == "__main__":

    board = CheckBoard(total_cells=100, interaction_step=3, start="popper", share_active=1.0,
                       start_proportion_intolerant=0.3, reinforce="when_intolerant", influence="drag_down")
    board.print_checkboard()
    board.interact_n_times(n_of_interactions=1000)
    board.print_checkboard()
