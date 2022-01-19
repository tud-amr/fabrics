import casadi as ca

def createMapping(phi, name, q, qdot):
    # differential map and jacobian phi : Q -> X
    J = ca.jacobian(phi, q)
    Jdot = ca.jacobian(ca.mtimes(J, qdot), q)
    phi_fun = ca.Function("phi_" + name, [q], [phi])
    J_fun = ca.Function("J_" + name, [q], [J])
    Jdot_fun = ca.Function("Jdot_" + name, [q, qdot], [Jdot])
    return (phi_fun, J_fun, Jdot_fun)


def createTimeVariantMapping(phi, name, q, qdot, t):
    # differential map and jacobian phi : Q -> X
    J = ca.jacobian(phi, q)
    Jdot = ca.jacobian(ca.mtimes(J, qdot), q)
    phi_fun = ca.Function("phi_" + name, [q, t], [phi])
    J_fun = ca.Function("J_" + name, [q, t], [J])
    Jdot_fun = ca.Function("Jdot_" + name, [q, qdot, t], [Jdot])
    return (phi_fun, J_fun, Jdot_fun)

def generateLagrangian(L, q, qdot, name):
    dL_dq = ca.gradient(L, q)
    dL_dqdot = ca.gradient(L, qdot)
    d2L_dq2 = ca.jacobian(dL_dq, q)
    d2L_dqdqdot = ca.jacobian(dL_dq, qdot)
    d2L_dqdot2 = ca.jacobian(dL_dqdot, qdot)

    M = d2L_dqdot2
    F = d2L_dqdqdot
    f_e = -dL_dq
    f = ca.mtimes(ca.transpose(F), qdot) + f_e
    return (M, f)

def generateHamiltonian(L, q, qdot, name):
    dL_dqdot = ca.gradient(L, qdot)
    He = ca.dot(dL_dqdot, qdot) - L
    return ca.Function("He_" + name, [q, qdot], [He])

def generateEnergizer(L, q, qdot, name, n, debug=False):
    (Me, fe) = generateLagrangian(L, q, qdot, name)
    h = ca.SX.sym("h", n)
    a1 = ca.dot(qdot, ca.mtimes(Me, qdot))
    a2 = ca.dot(qdot, ca.mtimes(Me, h) - fe)
    a = a2/(a1 + 1e-6)
    #a = a2/a1
    a_fun = ca.Function('a_' + name, [q, qdot, h], [a])
    if debug:
        a1_fun = ca.Function('a1_' + name, [q, qdot], [a1])
        a2_fun = ca.Function('a2_' + name, [q, qdot, h], [a2])
        M_fun = ca.Function('M_' + name, [q, qdot], [Me])
        f_fun = ca.Function('f_' + name, [q, qdot], [fe])
        return a_fun, a1_fun, a2_fun, M_fun, f_fun
    return a_fun
