from numphi import popper
from numphi.popper import Cell


def test_influence():

    influencer = Cell(t=0.0, a=1.0, d=1.0)
    influenced = Cell(t=1.0, a=1.0, d=0.99)

    new_influenced = popper.influence(influencer=influencer, influenced=influenced, direction="drag_down")

    assert new_influenced.t == 0.9

    influencer = Cell(t=0.99, a=1.0, d=1.0)
    influenced = Cell(t=1.0, a=1.0, d=0.99)

    assert popper.influence(influencer=influencer, influenced=influenced, direction="drag_down").t == 0.99

    influencer = Cell(t=0.0, a=1.0, d=1.0)
    influenced = Cell(t=1.0, a=1.0, d=1.0)

    assert popper.influence(influencer=influencer, influenced=influenced, direction="drag_down").t == 1.0


def test_reinforce():

    influencer = Cell(t=0.0, a=1.0, d=1.0)
    influenced = Cell(t=1.0, a=0.96, d=0.99)

    new_influenced = popper.reinforce(influencer=influencer, influenced=influenced, direction="always")

    assert new_influenced.a == 0.95
    assert new_influenced.d == 0.98

    influencer = Cell(t=0.9, a=0.2, d=1.0)
    influenced = Cell(t=1.0, a=0.2, d=0.99)

    assert popper.reinforce(influencer=influencer, influenced=influenced, direction="always").a == 0.21
    assert popper.reinforce(influencer=influencer, influenced=influenced, direction="always").d == 1.0

    influencer = Cell(t=1.0, a=0.0, d=0.01)
    influenced = Cell(t=0.0, a=0.01, d=0.5)

    assert popper.reinforce(influencer=influencer, influenced=influenced, direction="always").a == 0.0

    assert popper.reinforce(influencer=influencer, influenced=influenced, direction="always").d == 0.49

    # assert that iteration doesn't change input object unless reassigned

    d_before_iter = influenced.d

    popper.reinforce(influencer=influencer, influenced=influenced, direction="always")

    d_after_iter = influenced.d

    assert d_before_iter == d_after_iter


