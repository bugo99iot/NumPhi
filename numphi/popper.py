import numpy as np
import logging
import math as math
import random

from numphi.parameters import COLORS_ALLOWED, INFLUENCE_TYPE
from numphi.exceptions import BoardException, ActorException
from numphi.utils import is_square, print_checkboard

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


class Board(object):

    def __init__(self, n_actors: int, interaction_step: int = 1, color_gradient: tuple = ("red", "blue"),
                 start: str = "random", start_proportion_intolerant: float = None):

        if isinstance(n_actors, int) is False or n_actors < 9:

            raise BoardException("n_actors must be a square number integer >= 9")

        if is_square(n_actors) is False:
            raise BoardException("n_actors must be a square number")

        self.n_actors = n_actors

        self.board_side = int(math.sqrt(self.n_actors))

        if isinstance(interaction_step, int) is False or interaction_step < 1:
            raise BoardException("interaction_step must be an integer > 0")

        self.interaction_step = interaction_step

        if interaction_step > self.board_side - 1:

            logging.warning("Interaction step is large. Actors interac with full board.")

            self.interaction_step = self.board_side - 1

        if isinstance(color_gradient, tuple) is False or (False in [isinstance(k, str) for k in color_gradient]) \
                or (True in [k not in COLORS_ALLOWED for k in color_gradient]) or (len(color_gradient)) != 2 or \
                (color_gradient[0] == color_gradient[1]):

            raise BoardException("Color gradient must be a tuple of colors amongst allowed: {}".format(COLORS_ALLOWED))

        self.color_gradient = color_gradient

        self.start = start.lower() if start is not None else None

        self.start_proportion_intolerant = start_proportion_intolerant

        if self.start_proportion_intolerant is not None:

            if isinstance(self.start_proportion_intolerant, float) is False or (0.0 <= self.start_proportion_intolerant
                                                                                <= 1.0) is False:

                raise BoardException("start_proportion must be a number between 0.0 and 1.0")

        if start == "random":

            self.actors = [Actor(t=np.random.randint(0, 100)/100.0, a=np.random.randint(0, 100)/100.0,
                                 d=np.random.randint(0, 100)/100.0) for k in range(n_actors)]

        elif start == "popper":

            if start_proportion_intolerant is None:

                raise BoardException("If you choose Popper as start, you need to set start_proportion variable, "
                                     "e.g. start_proportion=0.9")

            n_intolerant = int(start_proportion_intolerant * self.n_actors)

            n_tolerant = self.n_actors - n_intolerant

            self.actors = [Actor(t=0.0, a=1.0, d=1.0)]*n_intolerant + [Actor(t=1.0, a=0.0, d=0.0)]*n_tolerant

            random.shuffle(self.actors, random.random)

    def print_checkboard(self):
        """
        Print current checkboard

        :return:
        """

        print_checkboard(actors_list=self.actors, colors=self.color_gradient)

        return None

    def print_distribution(self):
        """
        Print tolerance distribution in 0.1 buckets
        :return:
        """

        return None

    def interact_n_times(self, n_of_interactions: int = 1) -> None:

        print([k.t for k in self.actors])

        actors_after_interact = list()

        for steps in range(n_of_interactions):

            for index, item in enumerate(self.actors[1:-1]):

                #print("Before: {}".format(item.t))

                print("Influencer t: {}".format(self.actors[index+1].t))
                print("Influenced t: {}".format(item.t))


                item = influence(influenced=item, influencer=self.actors[index+1])
                #item = influence(influenced=item, influencer=self.actors[index-1])

                #print("After: {}".format(item.t))

                print("Influenced t after: {}".format(item.t))

                print()

                self.actors[index] = item

            print([k.t for k in self.actors])

            print_checkboard(actors_list=self.actors, colors=self.color_gradient)

        return None


class Actor(object):

    def __init__(self, t: float, a: float, d: float):

        self.t = t
        self.a = a
        self.d = d


def influence(influenced: Actor, influencer: Actor, direction: str = "lower") -> Actor:

    if direction not in INFLUENCE_TYPE:
        raise Exception("direction must be 'lower' or 'bi'")

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


if __name__ == "__main__":

    test_actor = Actor(t=1.0, a=0.2, d=0.1)

    #board = Board(n_actors=9, interaction_step=1, color_gradient=("red", "blue"))
    #board.print_checkboard

    board = Board(n_actors=81, interaction_step=1, color_gradient=("red", "blue"), start="popper", start_proportion_intolerant=0.5)
    #board.print_checkboard()
    board.interact_n_times(n_of_interactions=30)
    #board.print_checkboard()

    #board = Board(n_actors=25, interaction_step=1, color_gradient=("red", "blue"), start="random")
    #board.interact_n_times(n_of_interactions=1000)

    #influencer = Actor(t=0.0, a=1.0, d=1.0)
    #influenced = Actor(t=1.0, a=1.0, d=0.99)

    #influenced_after_influence = influence(influencer=influencer, influenced=influenced, direction="lower")
