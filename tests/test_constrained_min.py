import math
import unittest
import src.constrained_min as constrained_min
import src.utils as utils
import examples
import numpy as np
import matplotlib.pyplot as plt


class TestConstrainedOptimization(unittest.TestCase):

    def test_qp(self):
        f = lambda x, y: np.sqrt(x ** 2 + y ** 2) - 1

        surface = lambda x, y: 1 - x - y

        x = np.linspace(-0.5, 0.5, 100)
        y = np.linspace(-0.5, 0.5, 100)

        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('f(x, y) = np.sqrt(x ** 2 + y ** 2) - 1')

        # ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

        Z_surface = surface(X, Y)
        ax.plot_surface(X, Y, Z_surface, cmap='plasma', alpha=0.5)

        # ax.view_init(elev=10, azim=45)  # Set the elevation and azimuth angles

        example_name = '"constrained quadratic programming"'
        print(f'Testing {example_name}')

        x0 = np.array([0.1, 0.2, 0.7])

        constrained_qp_func = lambda t: (t[0] ** 2) + (t[1] ** 2) + (t[2] + 1) ** 2

        eq_constraints_mat = np.array([[1, 1, 1]])
        eq_constraints_rhs = np.array([1])
        c1 = lambda t: (-1) * t[0]
        c2 = lambda t: (-1) * t[1]
        c3 = lambda t: (-1) * t[2]
        ineq_constraints = [c1, c2, c3]

        x_final, f_final, iter_func_val = constrained_min.interior_pt(constrained_qp_func, ineq_constraints,
                                                                      eq_constraints_mat, x0)

        for point in iter_func_val:
            print(
                f'Iteration: {point[0]}; point=({point[1][0]},{point[1][1]},{point[1][2]}); function value={point[2]}')

        # Add final point
        # points = [x_final]
        # for point in points:
        #     ax.scatter(point[0], point[1], color='red', marker='o')

        # Add path points
        for point in iter_func_val:
            ax.scatter(point[1][0], point[1][1], color='red', marker='o')

        x1 = []
        x2 = []
        for point in iter_func_val:
            x1 += [point[0]]
            x2 += [point[2]]

        plt.figure()
        plt.plot(x1, x2, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.title(f'Function values at each iteration for {example_name}')

        plt.show()

        print()

    def test_lp(self):
        f = lambda x: -x

        x = np.linspace(-10, 10, 100)
        y = f(x)

        plt.figure()
        plt.plot(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('f(x) = -x')

        example_name = '"constrained linear programming"'
        print(f'Testing {example_name}')

        x0 = np.array([0.5, 0.75])

        constrained_lp_func = lambda t: (-1) * t[0] + (-1) * t[1]

        c1 = lambda t: (-1) * t[0] + 1 + (-1) * t[1]
        c2 = lambda t: t[1] - 1
        c3 = lambda t: t[0] - 2
        c4 = lambda t: (-1) * t[1]
        ineq_constraints = [c1, c2, c3, c4]

        x_final, f_final, iter_func_val = constrained_min.interior_pt(constrained_lp_func, ineq_constraints,
                                                                      None, x0)

        for point in iter_func_val:
            print(
                f'Iteration: {point[0]}; point=({point[1][0]},{point[1][1]}); function value={(-1)*point[2]}')

        # Add final point
        # plt.scatter(x_final[0], x_final[1], color='red', marker='o', label=f'({x_final[0]}, {x_final[1]})')
        # plt.text(x_final[0], x_final[1], f'({x_final[0]}, {x_final[1]})', verticalalignment='center')

        # Add path points
        for point in iter_func_val:
            plt.scatter(point[1][0], point[1][1], color='red', marker='o')

        x1 = []
        x2 = []
        for point in iter_func_val:
            x1 += [point[0]]
            x2 += [(-1)*point[2]]

        plt.figure()
        plt.plot(x1, x2, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.title(f'Function values at each iteration for {example_name}')

        plt.show()

        print()


if __name__ == '__main__':
    unittest.main()
