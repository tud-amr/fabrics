from col_avoidance import pointMassAvoidance
from col_avoidance_moving import pointMassDynamicAvoidance
from pointMass import pointMass
from pointMass_dynamic import pointMassDynamicGoal
from pointMass_moving import pointMassDynamicAnnoying


def test_pointMassAvoidance():
    res = pointMassAvoidance(10, render=False)
    assert isinstance(res, dict)


def test_pointMassDynamicAvoidance():
    res = pointMassDynamicAvoidance(10, render=False)
    assert isinstance(res, dict)


def test_pointMassDynamic():
    res = pointMassDynamicGoal(n_steps=10, render=False)
    assert isinstance(res, dict)


def test_pointMassSimple():
    res = pointMass(n_steps=10, render=False)
    assert isinstance(res, dict)


def test_pontMassDynamicgoal():
    res = pointMassDynamicAnnoying(n_steps=10, render=False)
    assert isinstance(res, dict)
