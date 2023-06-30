import math

import numpy as np
from autograd import grad, jacobian


def g(f, ineq_constraints, x, t):
    df = grad(f)
    df_x = df(x)
    df_x.shape = (df_x.shape[0], 1)
    s = np.zeros((x.shape[0], 1))
    for func in ineq_constraints:
        dfunc = grad(func)
        sfunc = 1 / (-1 * func(x)) * dfunc(x)
        sfunc.shape = (x.shape[0], 1)
        s += sfunc
    return t * df_x + s


def h(f, ineq_constraints, x, t):
    x = np.squeeze(x)
    df = grad(f)
    hf = jacobian(df)
    s1 = np.zeros((x.shape[0], x.shape[0]))
    for func in ineq_constraints:
        dfunc = grad(func)
        s1 += 1 / (func(x) ** 2) * dfunc(x).reshape(-1, 1) @ dfunc(x).reshape(-1, 1).T

    s2 = np.zeros((x.shape[0], x.shape[0]))
    for func in ineq_constraints:
        dfunc = grad(func)
        hfunc = jacobian(dfunc)
        s2 += 1 / (-1 * func(x)) * hfunc(x)

    return t * hf(x) + s1 + s2


def backtracking(c1, p, f, pk, xk, df_xk):
    alpha = 0.5
    xk.shape = (xk.shape[0], 1)
    try:
        wolfe_conditions_met = f(xk + alpha * pk)[0] <= f(xk)[0] + c1 * alpha * (df_xk.T @ pk)
    except Exception:
        wolfe_conditions_met = False
    while not wolfe_conditions_met:
        alpha = p * alpha
        try:
            wolfe_conditions_met = f(xk + alpha * pk)[0] <= f(xk)[0] + c1 * alpha * (df_xk.T @ pk)
        except Exception:
            wolfe_conditions_met = False

    return alpha


def newton_equality_constrained(f_orig, ineq_constraints, A, x0, t, phi, e):
    max_iter = 1000
    f = lambda x: t * f_orig(x) + phi(x)
    x_prev = x0
    i = 0
    while i < max_iter:
        i = i + 1
        df_prev = g(f_orig, ineq_constraints, x_prev, t)
        df_prev.shape = (df_prev.shape[0], 1)
        hf_prev = h(f_orig, ineq_constraints, x_prev, t)

        if A is None:
            try:
                inv_h = np.linalg.inv(hf_prev)
            except Exception:
                print()
            delta = -inv_h.dot(df_prev)
            p_nt = delta
            lambda_prev = np.sqrt(p_nt.T @ hf_prev @ p_nt)
        else:
            system_matrix = np.block([[hf_prev, A.T], [A, np.zeros((1, 1))]])
            system_rhs = np.concatenate([-df_prev, np.zeros((1, 1))])
            delta_x = np.linalg.solve(system_matrix, system_rhs)
            p_nt = delta_x[:len(x0)]
            lambda_prev = np.sqrt(p_nt.T @ hf_prev @ p_nt)

        if 0.5 * lambda_prev ** 2 < e:
            return x_prev
        else:
            c1 = 0.01
            p = 0.5
            step_length = backtracking(c1, p, f, p_nt, x_prev, df_prev)
            if step_length is not None:
                x_next = x_prev + step_length * p_nt
                x_next = np.squeeze(x_next)
                x_prev = x_next
            else:
                print('ERROR')

    return x_prev


def log_barrier_method(f, ineq_constraints, eq_constraints_mat, x0, t, phi, m, mu, e=1e-5):
    x = newton_equality_constrained(f, ineq_constraints, eq_constraints_mat, x0, t, phi, e)
    iter_func_val = []
    i = 0
    iter_func_val.append((i, x, f(x)))
    while m / t >= e:
        i = i + 1
        t = mu * t
        x = newton_equality_constrained(f, ineq_constraints, eq_constraints_mat, x, t, phi, e)
        iter_func_val.append((i, x, f(x)))

    return x, f(x), iter_func_val


def interior_pt(f, ineq_constraints, eq_constraints_mat, x0, t=1):
    def phi(x):
        s = 0
        for func in ineq_constraints:
            s += math.log(-1 * func(x))
        return -1 * s

    return log_barrier_method(f, ineq_constraints, eq_constraints_mat, x0, t, phi, m=len(ineq_constraints), mu=10)
