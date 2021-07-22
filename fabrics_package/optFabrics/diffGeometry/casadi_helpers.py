import casadi as ca


def outerProduct(a: ca.SX, b: ca.SX):
    assert isinstance(a, ca.SX)
    assert isinstance(b, ca.SX)
    m = a.size()[0]
    A = ca.transpose(ca.repmat(ca.transpose(a), m))
    B = ca.repmat(ca.transpose(b), m)
    return ca.times(A, B)
