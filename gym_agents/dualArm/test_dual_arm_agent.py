from dualArmAgent import dualArmFabric

def test_dualArmFabric():
    res = dualArmFabric(n_steps=10, render=False)
    assert isinstance(res, dict)
