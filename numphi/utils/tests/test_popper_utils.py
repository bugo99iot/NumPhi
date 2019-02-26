from numphi.utils.popper_utils import get_coordinates_of_all_cells
from collections import Counter


# Counter(get_all_combos_with_step(coords=(0, 0),
# interaction_step=1, board_side=2)) == Counter([(1, 1), (0, 1), (1, 0)])

def test_get_coordinates_of_all_cells():

    assert get_coordinates_of_all_cells(board_side=2) == [(0, 0), (0, 1), (1, 0), (1, 1)]
