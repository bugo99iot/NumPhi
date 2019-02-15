from numphi.utils.popper_utils import get_all_combos
from collections import Counter


def test_get_get_all_combos():

    assert Counter(get_all_combos(coords=(0, 0), interaction_step=1, board_side=2)) == Counter([(1, 1), (0, 1), (1, 0)])
