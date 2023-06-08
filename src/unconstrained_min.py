import numpy as np


def backtracking(c1, p, f, pk, xk, df_xk):
    alpha = 1
    wolfe_conditions_met = f(xk + alpha * pk)[0] <= f(xk)[0] + c1 * alpha * pk.dot(np.transpose(df_xk))
    while not wolfe_conditions_met:
        alpha = p * alpha
        wolfe_conditions_met = f(xk + alpha * pk)[0] <= f(xk)[0] + c1 * alpha * pk.dot(np.transpose(df_xk))

    return alpha


def gradient_descent(f, x0, obj_tol, param_tol, max_iter, c1, p):
    iter_func_val = []
    x_prev = x0
    f_prev, _ = f(x_prev, need_hessian_eval=False)
    i = 0
    iter_func_val.append((i, x_prev, f_prev))
    success = False
    while i < max_iter:
        i = i + 1
        f_prev, df_prev = f(x_prev, need_hessian_eval=False)
        print(f'Iteration number: {i}; Current location: {x_prev}; Current objective value: {f_prev}')
        delta = - df_prev / np.linalg.norm(df_prev)
        step_length = backtracking(c1, p, f, delta, x_prev, df_prev)
        if step_length is not None:
            x_next = x_prev + step_length * delta
            f_next, _ = f(x_next, need_hessian_eval=False)
            iter_func_val.append((i, x_next, f_next))
        if abs(f_next - f_prev) < obj_tol or np.linalg.norm(x_next - x_prev) < param_tol:
            success = True
            return x_next, f_next, success, iter_func_val
        else:
            x_prev = x_next

    return x_next, f_next, success, iter_func_val


def newton_descent(f, x0, obj_tol, param_tol, max_iter, c1, p):
    iter_func_val = []
    x_prev = x0
    f_prev, _ = f(x_prev, need_hessian_eval=False)
    i = 0
    iter_func_val.append((i, x_prev, f_prev))
    success = False
    while i < max_iter:
        i = i + 1
        f_prev, df_prev, hf_prev = f(x_prev, need_hessian_eval=True)
        print(f'Iteration number: {i}; Current location: {x_prev}; Current objective value: {f_prev}')
        try:
            inv_h = np.linalg.inv(hf_prev)
        except Exception:
            return None, None, success, None

        delta = -inv_h.dot(df_prev)
        step_length = backtracking(c1, p, f, delta, x_prev, df_prev)
        if step_length is not None:
            x_next = x_prev + step_length * delta
            f_next, _ = f(x_next, need_hessian_eval=False)
            iter_func_val.append((i, x_next, f_next))
        if abs(f_next - f_prev) < obj_tol or np.linalg.norm(x_next - x_prev) < param_tol:
            success = True

            return x_next, f_next, success, iter_func_val
        else:
            x_prev = x_next

    return x_next, f_next, success, iter_func_val


def bfgs_descent(f, x0, obj_tol, param_tol, max_iter, c1, p):
    iter_func_val = []
    x_prev = x0
    f_prev, _, hf_prev = f(x_prev, need_hessian_eval=True)
    i = 0
    iter_func_val.append((i, x_prev, f_prev))
    bk = hf_prev
    success = False
    while i < max_iter:
        i = i + 1
        f_prev, df_prev = f(x_prev, need_hessian_eval=False)
        print(f'Iteration number: {i}; Current location: {x_prev}; Current objective value: {f_prev}')
        try:
            inv_bk = np.linalg.inv(bk)
        except Exception:
            return None, None, success, None

        delta = -inv_bk.dot(df_prev)
        step_length = backtracking(c1, p, f, delta, x_prev, df_prev)
        if step_length is not None:
            x_next = x_prev + step_length * delta
            f_next, df_next = f(x_next, need_hessian_eval=False)
            iter_func_val.append((i, x_next, f_next))
            sk = (x_next - x_prev)
            sk.shape = (2, 1)
            yk = (df_next - df_prev)
            yk.shape = (2, 1)
            # if going to divide by zero. use hessian
            if (np.transpose(sk) @ bk @ sk)[0][0] == 0 or (np.transpose(yk) @ sk)[0][0] == 0:
                bk_1 = f(x_next, need_hessian_eval=True)[2]
            else:
                bk_1 = bk - ((bk @ sk @ np.transpose(sk) @ bk)/(np.transpose(sk) @ bk @ sk)) + ((yk @ np.transpose(yk))/(np.transpose(yk) @ sk))
            bk = bk_1
        if abs(f_next - f_prev) < obj_tol or np.linalg.norm(x_next - x_prev) < param_tol:
            success = True

            return x_next, f_next, success, iter_func_val
        else:
            x_prev = x_next

    return x_next, f_next, success, iter_func_val


def sr1_descent(f, x0, obj_tol, param_tol, max_iter, c1, p):
    iter_func_val = []
    x_prev = x0
    f_prev, _, hf_prev = f(x_prev, need_hessian_eval=True)
    i = 0
    iter_func_val.append((i, x_prev, f_prev))
    bk = hf_prev
    success = False
    while i < max_iter:
        i = i + 1
        f_prev, df_prev = f(x_prev, need_hessian_eval=False)
        print(f'Iteration number: {i}; Current location: {x_prev}; Current objective value: {f_prev}')
        try:
            inv_bk = np.linalg.inv(bk)
        except Exception:
            return None, None, success, None

        delta = -inv_bk.dot(df_prev)
        step_length = backtracking(c1, p, f, delta, x_prev, df_prev)
        if step_length is not None:
            x_next = x_prev + step_length * delta
            f_next, df_next = f(x_next, need_hessian_eval=False)
            iter_func_val.append((i, x_next, f_next))
            sk = (x_next - x_prev)
            sk.shape = (2, 1)
            yk = (df_next - df_prev)
            yk.shape = (2, 1)
            # if going to divide by zero. use hessian
            if ((yk - bk @ sk).T @ sk)[0][0] == 0:
                bk_1 = f(x_next, need_hessian_eval=True)[2]
            else:
                bk_1 = bk + (((yk - bk @ sk) @ (yk - bk @ sk).T)/((yk - bk @ sk).T @ sk))
            bk = bk_1
        if abs(f_next - f_prev) < obj_tol or np.linalg.norm(x_next - x_prev) < param_tol:
            success = True

            return x_next, f_next, success, iter_func_val
        else:
            x_prev = x_next

    return x_next, f_next, success, iter_func_val