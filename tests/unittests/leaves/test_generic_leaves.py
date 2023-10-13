import casadi as ca
import numpy as np
from fabrics.diffGeometry.diffMap import DynamicDifferentialMap
from fabrics.helpers.variables import Variables
from fabrics.components.leaves.dynamic_leaf import DynamicLeaf
from fabrics.components.leaves.leaf import Leaf


def test_dynamic_leaf_generation():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    root_variables = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    phi_leaf = q
    dynamic_leaf = DynamicLeaf(root_variables, "dynamic_leaf", phi_leaf, dim=2)
    dynamic_map = dynamic_leaf.dynamic_map()
    assert isinstance(dynamic_map, DynamicDifferentialMap)

def test_leaf_generation():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    root_variables = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    phi_leaf = q
    static_leaf = Leaf(root_variables, "static_leaf", phi_leaf, dim=2)
    static_map = static_leaf.map()
    assert static_map
    static_leaf.concretize()
    leaf_eval = static_leaf.evaluate(q=np.array([0.1, 0.1]), qdot=np.array([0.3, -0.7]))
    assert leaf_eval['x'][0] == 0.1








