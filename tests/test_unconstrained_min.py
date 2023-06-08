import unittest
import src.unconstrained_min as unconstrained_min
import src.utils as utils
import examples
import numpy as np


class TestLineSearchMinimization(unittest.TestCase):

    def test_quadratic1_func(self):
        example_name = '"quadratic 1"'
        print(f'Testing {example_name}')
        utils.create_plot(examples.quadratic1_func, x1_min=-3, x1_max=3, x2_min=-3, x2_max=3, levels=50, example_name=example_name)

        x0 = np.array([1., 1.])
        obj_tol = 10 ** -8
        param_tol = 10 ** -12
        max_iter = 100
        c1 = 0.01
        p = 0.5

        execute_funcs = [unconstrained_min.gradient_descent, unconstrained_min.newton_descent,
                         unconstrained_min.bfgs_descent, unconstrained_min.sr1_descent]
        successful_funcs_names = []
        successful_funcs_data = []

        for func in execute_funcs:
            print(f'========= {func.__name__} =========')
            x_final, f_final, success, iter_func_val = func(examples.quadratic1_func, x0, obj_tol, param_tol, max_iter, c1, p)

            print(f'x_final: {x_final}, f_final: {f_final}, success: {success}')

            if iter_func_val is not None:
                successful_funcs_names.append(func.__name__)
                successful_funcs_data.append(iter_func_val)

        utils.plot_path(successful_funcs_names, successful_funcs_data)
        utils.plot_values(successful_funcs_names, successful_funcs_data, example_name)

    def test_quadratic2_func(self):
        example_name = '"quadratic 2"'
        print(f'Testing {example_name}')
        utils.create_plot(examples.quadratic2_func, x1_min=-3, x1_max=3, x2_min=-3, x2_max=3, levels=50, example_name=example_name)

        x0 = np.array([1., 1.])
        obj_tol = 10 ** -8
        param_tol = 10 ** -12
        max_iter = 100
        c1 = 0.01
        p = 0.5

        execute_funcs = [unconstrained_min.gradient_descent, unconstrained_min.newton_descent,
                         unconstrained_min.bfgs_descent, unconstrained_min.sr1_descent]
        successful_funcs_names = []
        successful_funcs_data = []

        for func in execute_funcs:
            print(f'========= {func.__name__} =========')
            x_final, f_final, success, iter_func_val = func(examples.quadratic2_func, x0, obj_tol, param_tol, max_iter, c1, p)

            print(f'x_final: {x_final}, f_final: {f_final}, success: {success}')

            if iter_func_val is not None:
                successful_funcs_names.append(func.__name__)
                successful_funcs_data.append(iter_func_val)

        utils.plot_path(successful_funcs_names, successful_funcs_data)
        utils.plot_values(successful_funcs_names, successful_funcs_data, example_name)

    def test_quadratic3_func(self):
        example_name = '"quadratic 3"'
        print(f'Testing {example_name}')
        utils.create_plot(examples.quadratic3_func, x1_min=-3, x1_max=3, x2_min=-3, x2_max=3, levels=50, example_name=example_name)

        x0 = np.array([1., 1.])
        obj_tol = 10 ** -8
        param_tol = 10 ** -12
        max_iter = 100
        c1 = 0.01
        p = 0.5

        execute_funcs = [unconstrained_min.gradient_descent, unconstrained_min.newton_descent,
                         unconstrained_min.bfgs_descent, unconstrained_min.sr1_descent]
        successful_funcs_names = []
        successful_funcs_data = []

        for func in execute_funcs:
            print(f'========= {func.__name__} =========')
            x_final, f_final, success, iter_func_val = func(examples.quadratic3_func, x0, obj_tol, param_tol, max_iter, c1, p)

            print(f'x_final: {x_final}, f_final: {f_final}, success: {success}')

            if iter_func_val is not None:
                successful_funcs_names.append(func.__name__)
                successful_funcs_data.append(iter_func_val)

        utils.plot_path(successful_funcs_names, successful_funcs_data)
        utils.plot_values(successful_funcs_names, successful_funcs_data, example_name)

    def test_rosenbrock_func(self):
        example_name = '"Rosenbrock function"'
        print(f'Testing {example_name}')
        utils.create_plot(examples.rosenbrock_func, x1_min=-3, x1_max=3, x2_min=-3, x2_max=3, levels=50, example_name=example_name)

        x0 = np.array([-1., 2.])
        obj_tol = 10 ** -8
        param_tol = 10 ** -12
        max_iter = 10000
        c1 = 0.01
        p = 0.5

        execute_funcs = [unconstrained_min.gradient_descent, unconstrained_min.newton_descent,
                         unconstrained_min.bfgs_descent, unconstrained_min.sr1_descent]
        successful_funcs_names = []
        successful_funcs_data = []

        for func in execute_funcs:
            print(f'========= {func.__name__} =========')
            x_final, f_final, success, iter_func_val = func(examples.rosenbrock_func, x0, obj_tol, param_tol, max_iter, c1, p)

            print(f'x_final: {x_final}, f_final: {f_final}, success: {success}')

            if iter_func_val is not None:
                successful_funcs_names.append(func.__name__)
                successful_funcs_data.append(iter_func_val)

        utils.plot_path(successful_funcs_names, successful_funcs_data)
        utils.plot_values(successful_funcs_names, successful_funcs_data, example_name)

    def test_linear_func(self):
        example_name = '"linear function"'
        print(f'Testing {example_name}')
        utils.create_plot(examples.linear_func, x1_min=-1, x1_max=1, x2_min=-1, x2_max=1, levels=50, example_name=example_name)

        x0 = np.array([1., 1.])
        obj_tol = 10 ** -8
        param_tol = 10 ** -12
        max_iter = 100
        c1 = 0.01
        p = 0.5

        execute_funcs = [unconstrained_min.gradient_descent, unconstrained_min.newton_descent,
                         unconstrained_min.bfgs_descent, unconstrained_min.sr1_descent]
        successful_funcs_names = []
        successful_funcs_data = []

        for func in execute_funcs:
            print(f'========= {func.__name__} =========')
            x_final, f_final, success, iter_func_val = func(examples.linear_func, x0, obj_tol, param_tol, max_iter, c1, p)

            print(f'x_final: {x_final}, f_final: {f_final}, success: {success}')

            if iter_func_val is not None:
                successful_funcs_names.append(func.__name__)
                successful_funcs_data.append(iter_func_val)

        utils.plot_path(successful_funcs_names, successful_funcs_data)
        utils.plot_values(successful_funcs_names, successful_funcs_data, example_name)

    def test_g_func(self):
        example_name = '"smoothed corner triangles"'
        print(f'Testing {example_name}')
        utils.create_plot(examples.g_func, x1_min=-1, x1_max=1, x2_min=-1, x2_max=1, levels=50, example_name=example_name)

        x0 = np.array([1., 1.])
        obj_tol = 10 ** -8
        param_tol = 10 ** -12
        max_iter = 100
        c1 = 0.01
        p = 0.5

        execute_funcs = [unconstrained_min.gradient_descent, unconstrained_min.newton_descent,
                         unconstrained_min.bfgs_descent, unconstrained_min.sr1_descent]
        successful_funcs_names = []
        successful_funcs_data = []

        for func in execute_funcs:
            print(f'========= {func.__name__} =========')
            x_final, f_final, success, iter_func_val = func(examples.g_func, x0, obj_tol, param_tol, max_iter, c1, p)

            print(f'x_final: {x_final}, f_final: {f_final}, success: {success}')

            if iter_func_val is not None:
                successful_funcs_names.append(func.__name__)
                successful_funcs_data.append(iter_func_val)

        utils.plot_path(successful_funcs_names, successful_funcs_data)
        utils.plot_values(successful_funcs_names, successful_funcs_data, example_name)


if __name__ == '__main__':
    unittest.main()
