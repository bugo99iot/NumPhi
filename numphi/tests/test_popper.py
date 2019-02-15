from numphi import popper
from numphi.popper import Cell


def test_influence():

    influencer = Cell(t=0.0, a=1.0, d=1.0)
    influenced = Cell(t=1.0, a=1.0, d=0.99)

    new_influenced = popper.influence(influencer=influencer, influenced=influenced, direction="lower")

    assert new_influenced.t == 0.9

    influencer = Cell(t=0.99, a=1.0, d=1.0)
    influenced = Cell(t=1.0, a=1.0, d=0.99)

    assert popper.influence(influencer=influencer, influenced=influenced, direction="lower").t == 0.99

    influencer = Cell(t=0.0, a=1.0, d=1.0)
    influenced = Cell(t=1.0, a=1.0, d=1.0)

    assert popper.influence(influencer=influencer, influenced=influenced, direction="lower").t == 1.0
