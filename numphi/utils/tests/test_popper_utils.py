from numphi.utils import popper_utils
from collections import Counter
from unittest import TestCase
import numpy as np


# Counter(get_all_combos_with_step(coords=(0, 0),
# interaction_step=1, board_side=2)) == Counter([(1, 1), (0, 1), (1, 0)])

#def test_get_coordinates_of_all_cells():

#    assert get_coordinates_of_all_cells(board_side=2) == [(0, 0), (0, 1), (1, 0), (1, 1)]

class TestPopperUtils(TestCase):

    def test_interaction_matrix(self):
        print("HERE:", popper_utils.build_interaction_matrix(friend_cells=3,
                                                             share_active=1.0, board_side=81)[0, 0])

        print("HERE:", popper_utils.build_interaction_matrix(friend_cells=3,
                                                             share_active=1.0, board_side=81)[2, 1])

        friend_coords = popper_utils.build_interaction_matrix(friend_cells=3, share_active=1.0,
                                                              board_side=81)[0, 0]

        for i in (0, 1), (1, 0):
            assert i in friend_coords

    def test_generate_n_friends(self):
        print("here2", list(popper_utils.generate_n_friends(center=(0, 2), board_side=3)))

    def test_get_all_neighboaurs_for_cell(self):

        print("here3", popper_utils.get_all_neighbours_for_cell(coords=(0, 2), board_side=3))
