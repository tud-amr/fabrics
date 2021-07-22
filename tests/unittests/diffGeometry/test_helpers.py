import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.casadi_helpers import outerProduct


def test_outer_product():
    a = ca.SX.sym("a", 2)
    b = ca.SX.sym("b", 2)
    o = outerProduct(a, b)
    o_fun = ca.Function("o", [a, b], [o])
    a_c = np.array([1.0, 2.0])
    b_c = np.array([-0.1, 0.6])
    res = np.array(o_fun(a_c, b_c))
    assert res[0, 0] == -0.1
    assert res[0, 1] == 0.6
    assert res[1, 0] == -0.2
    assert res[1, 1] == 1.2
