from numphi.utils.popper_utils import get_all_combos
from collections import Counter


def test_get_get_all_combos():

    assert Counter(get_all_combos(coords=(0, 0), interaction_step=1, board_side=2)) == Counter([(1, 1), (0, 1), (1, 0)])
    assert Counter(get_all_combos(coords=(1, 1), interaction_step=2, board_side=3)) == Counter([(0, 0),
                                                                                                (0, 1),
                                                                                                (1, 0),
                                                                                                (2, 2),
                                                                                                (1, 2),
                                                                                                (2, 1),
                                                                                                (0, 2),
                                                                                                (2, 0)])
