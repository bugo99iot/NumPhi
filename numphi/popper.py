import numpy as np
import logging
import math as math
import random

from numphi.parameters import COLORS_ALLOWED, INFLUENCE_TYPE
from numphi.exceptions import CheckBoardException, CellException
from numphi.utils import is_square, print_checkboard, get_all_combos

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

class CheckBoard(object):

    def __init__(self, n_cells: int, interaction_step: int = 1, color_gradient: tuple = ("red", "blue"),
                 start: str = "random", start_proportion_intolerant: float = None, reinforce: float = 0.0,
                 share_active: float = 1.0):

        # todo if opinion are similar, increase a and d

        if isinstance(n_cells, int) is False or n_cells < 9 or n_cells > 1000000:

            raise CheckBoardException("n_cells must be a square number integer n so that 9 <= n <= 1,000,000")

        if is_square(n_cells) is False:
            raise CheckBoardException("n_cells must be a square number")

        self.n_cells = n_cells

        if share_active is None or isinstance(share_active, float) is None \
                or (0.0 <= share_active <= 1.0) is False:

            raise CheckBoardException("reinforce must be a float between 0 and 1")

        self.share_active = share_active

        self.board_side = int(math.sqrt(self.n_cells))

        if isinstance(interaction_step, int) is False or interaction_step < 1:
            raise CheckBoardException("interaction_step must be an integer > 0")

        self.interaction_step = interaction_step

        if interaction_step > self.board_side - 1:

            logging.warning("Interaction step is large. cells interac with full board.")

            self.interaction_step = self.board_side - 1

        if reinforce is None or isinstance(reinforce, float) is None \
                or (0.0 <= reinforce <= 1.0) is False:

            raise CheckBoardException("reinforce must be a float between 0 and 1")

        self.reinforce = reinforce

        if isinstance(color_gradient, tuple) is False or (False in [isinstance(k, str) for k in color_gradient]) \
                or (True in [k not in COLORS_ALLOWED for k in color_gradient]) or (len(color_gradient)) != 2 or \
                (color_gradient[0] == color_gradient[1]):

            raise CheckBoardException("Color gradient must be a tuple of colors amongst allowed: {}".format(COLORS_ALLOWED))

        self.color_gradient = color_gradient

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

        print_checkboard(checkboard=self.checkboard, colors=self.color_gradient)

        return None

    def print_distribution(self):
        """
        Print tolerance distribution in 0.1 buckets
        :return:
        """

        return None

    def interact_n_times(self, n_of_interactions: int = 1) -> None:

        for steps in range(n_of_interactions):

            new_board = np.array([Cell(t=0.0, a=0.0, d=0.0)
                                  for _ in range(self.n_cells)]).reshape(self.board_side, self.board_side)

            for coords, cell in np.ndenumerate(self.checkboard):

                # find all cells being infulenced by current cell

                all_combos = get_all_combos(coords=coords, range=self.interaction_step, board_side=self.board_side)

                if self.share_active < 1.0:

                    # todo: make elegant
                    cut_index = random.randint(0, int(round(self.share_active*100.0))) / 100.0 * len(all_combos)

                    all_combos = all_combos[:round(int(cut_index))]

                for combo in all_combos:

                    influenced = influence(influenced=self.checkboard[combo], influencer=cell, direction="bi")

                    influenced = reinforce(influenced=influenced, influencer=cell, direction="bi")

                    new_board[combo] = influenced

            self.checkboard = new_board

            self.print_checkboard()

        return None


class Cell(object):

    def __init__(self, t: float, a: float, d: float, r: int = 1):
        """

        :param t: tolerance
        :param a: attack
        :param d: defense
        :param r: range
        """

        self.t = t
        self.a = a
        self.d = d
        self.r = r


def influence(influenced: Cell, influencer: Cell, direction: str = "lower", reinforce: float or None = None) -> Cell:

    if direction not in INFLUENCE_TYPE:
        raise Exception("direction must be 'lower' or 'bi'")

    # if two cells are of same opinion, tolerance stays the same and attack and defence of influenced increase

    if influencer.a > influenced.d:

        if influencer.t < influenced.t:

            amount = round((influenced.t - influencer.t)/10.0, 2)

            if amount < 0.01:

                amount = 0.01

            influenced.t -= amount

            influenced.t = round(influenced.t, 2)

            if influenced.t > 1.0:

                influenced.t = 1.0

            if influenced.t < 0.0:

                influenced.t = 0.0

        else:

            if direction == "bi":

                amount = round((influencer.t - influenced.t) / 10.0, 2)

                if amount < 0.01:
                    amount = 0.01

                influenced.t += amount

                influenced.t = round(influenced.t, 2)

                if influenced.t > 1.0:
                    influenced.t = 1.0

                if influenced.t < 0.0:
                    influenced.t = 0.0

    return influenced


def reinforce(influenced: Cell, influencer: Cell, direction: str = "lower") -> Cell:

    if direction not in INFLUENCE_TYPE:
        raise Exception("direction must be 'lower' or 'bi'")

    # if two cells are of same opinion, tolerance stays the same and attack and defence of influenced increase

    if abs(influencer.t - influenced.t) < 0.1:

        influenced.a += 0.01

        influenced.d += 0.01

        influenced.a = round(influenced.a, 2)

        influenced.d = round(influenced.d, 2)

        if influenced.a < 0.0:

            influenced.a = 0.0

        if influenced.a > 1.0:

            influenced.a = 1.0

        if influenced.d < 0.0:

            influenced.d = 0.0

        if influenced.d > 1.0:

            influenced.d = 1.0

    if direction == "bi":

        if abs(influencer.t - influenced.t) > 0.1:

            influenced.a -= 0.01

            influenced.d -= 0.01

            influenced.a = round(influenced.a, 2)

            influenced.d = round(influenced.d, 2)

            if influenced.a < 0.0:

                influenced.a = 0.0

            if influenced.a > 1.0:

                influenced.a = 1.0

            if influenced.d < 0.0:

                influenced.d = 0.0

            if influenced.d > 1.0:

                influenced.d = 1.0

    return influenced


if __name__ == "__main__":


    # board = Board(n_cells=9, interaction_step=1, color_gradient=("red", "blue"))
    # board.print_checkboard

    board = CheckBoard(n_cells=16, interaction_step=3, color_gradient=("red", "blue"), start="random",
                       start_proportion_intolerant=0.9, share_active=0.5)
    board.print_checkboard()
    board.interact_n_times(n_of_interactions=60)
    # board.print_checkboard()

    # board = Board(n_cells=25, interaction_step=1, color_gradient=("red", "blue"), start="random")
    # board.interact_n_times(n_of_interactions=1000)

    # influencer = cell(t=0.0, a=1.0, d=1.0)
    # influenced = cell(t=1.0, a=1.0, d=0.99)

    # influenced_after_influence = influence(influencer=influencer, influenced=influenced, direction="lower")
