from numphi.utils.popper_utils import get_all_combos_with_step, get_coordinates_of_all_cells
from collections import Counter


def test_get_get_all_combos():

    assert Counter(get_all_combos_with_step(coords=(0, 0), interaction_step=1, board_side=2)) == Counter([(1, 1),
                                                                                                          (0, 1),
                                                                                                          (1, 0)])

    assert Counter(get_all_combos_with_step(coords=(1, 1), interaction_step=2, board_side=3)) == Counter([(0, 0),
                                                                                                          (0, 1),
                                                                                                          (1, 0),
                                                                                                          (2, 2),
                                                                                                          (1, 2),
                                                                                                          (2, 1),
                                                                                                          (0, 2),
                                                                                                          (2, 0)])


def test_get_coordinates_of_all_cells():

    assert get_coordinates_of_all_cells(board_side=2) == [(0, 0), (0, 1), (1, 0), (1, 1)]
