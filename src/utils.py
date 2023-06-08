import numpy as np
import matplotlib.pyplot as plt


def create_plot(f, x1_min, x1_max, x2_min, x2_max, levels, example_name):
    x1 = np.linspace(x1_min, x1_max)
    x2 = np.linspace(x2_min, x2_max)
    z = np.zeros(([len(x1), len(x2)]))
    for i in range(0, len(x1)):
        for j in range(0, len(x2)):
            fv, g = f(np.array([x1[i], x2[j]]))
            z[j, i] = fv

    contours = plt.contour(x1, x2, z, levels, cmap=plt.cm.gnuplot)
    plt.clabel(contours, inline=1, fontsize=10)

    plt.xlabel('$x1$ ->')
    plt.ylabel('$x2$ ->')
    plt.title(f'Contour lines of function {example_name} with {levels} levels')


def plot_path(funcs_names, funcs_data):
    colormap = plt.colormaps.get_cmap('tab10')

    for i, points in enumerate(funcs_data):
        x1 = []
        x2 = []
        for point in points:
            x1 += [point[1][0], ]
            x2 += [point[1][1], ]
        color = colormap(i)
        plt.plot(x1, x2, color=color, marker='x', label=funcs_names[i])

    plt.legend()


def plot_values(funcs_names, funcs_data, example_name):
    fig, ax = plt.subplots()
    colormap = plt.colormaps.get_cmap('tab10')

    for i, points in enumerate(funcs_data):
        x1 = []
        x2 = []
        for point in points:
            x1 += [point[0]]
            x2 += [point[2]]
        color = colormap(i)
        ax.plot(x1, x2, color=color, marker='x', label=funcs_names[i])

    ax.set_xlabel('$x1$ ->')
    ax.set_ylabel('$x2$ ->')
    ax.set_title(f'Function values at each iteration for {example_name}')

    ax.legend()
    plt.show()
