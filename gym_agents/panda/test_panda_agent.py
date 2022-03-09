from pandaAgent import pandaFabric

def test_pandaFabric():
    res = pandaFabric(n_steps=10, render=False)
    assert isinstance(res, dict)
