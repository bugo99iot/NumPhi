import numpy as np
from numphi.exceptions import BoardException, ActorException
from numphi.utils import is_square
import logging
import math as math

import matplotlib.pyplot as plt

from numphi.parameters import COLORS_ALLOWED, INFLUENCE_TYPE

from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())

# define logging level
logger = logging.getLogger(__name__)

if os.getenv('ENV') in ['staging', 'production']:
    logging.basicConfig(level=logging.WARNING)
else:
    logging.basicConfig(level=logging.DEBUG)


# todo: add Sentry hook


class Board(object):

    def __init__(self, n_actors: int, interaction_step: int = 1, color_gradient: tuple = ("red", "blue"),
                 start: str = "random"):

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

        if start == "random":

            self.actors = [Actor(t=np.random.randint(0, 10)/1000.0, a=0.2, d=0.3) for k in range(n_actors)]


    @property
    def print_it(self):

        actors_to_print = np.asarray([k.t for k in self.actors])

        image = actors_to_print.reshape((self.board_side, self.board_side))

        plt.matshow(image)

        plt.show()

    def interact_n_times(self, n_of_interactions: int= 1):

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

            influenced.t -= influencer.t - influenced.t

        else:

            if direction == "bi":

                influenced.t += influencer.t - influenced.t

    return influenced


test_actor = Actor(t=1.0, a=0.2, d=0.1)

print(type(test_actor))


print(test_actor.t)

board = Board(n_actors=900, interaction_step=1, color_gradient=("red", "blue"))
board.print_it
