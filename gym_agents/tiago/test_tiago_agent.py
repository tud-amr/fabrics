from tiagoAgent import tiagoFabric

def test_tiagoFabric():
    res = tiagoFabric(n_steps=10, render=False)
    assert isinstance(res, dict)
