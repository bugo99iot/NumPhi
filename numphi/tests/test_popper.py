from numphi import popper
from numphi.popper import Actor


def test_influence():

    influencer = Actor(t=0.0, a=1.0, d=1.0)
    influenced = Actor(t=1.0, a=1.0, d=0.99)

    assert popper.influence(influencer=influencer, influenced=influenced, direction="lower").t == 0.9

    influencer = Actor(t=0.99, a=1.0, d=1.0)
    influenced = Actor(t=1.0, a=1.0, d=0.99)

    assert popper.influence(influencer=influencer, influenced=influenced, direction="lower").t == 0.99

    influencer = Actor(t=0.0, a=1.0, d=1.0)
    influenced = Actor(t=1.0, a=1.0, d=1.0)

    assert popper.influence(influencer=influencer, influenced=influenced, direction="lower").t == 1.0

