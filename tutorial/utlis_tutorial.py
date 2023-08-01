import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 200  # figure dpi


# animation
def update(num, lines, points, coords):
    start = max(0, num - 100)
    outputs = []
    for i in range(len(lines)):
        lines[i].set_data(coords[i]['x'][start:num], coords[i]['y'][start:num])
        points[i].set_data(coords[i]['x'][num], coords[i]['y'][num])
        outputs.append(lines[i])
        outputs.append(points[i])
    return outputs


def plotEnergies(energies, ax, t, equal=False, name=None):
    if equal:
        ax.plot(t, energies)
        ax.axis("equal")
    elif name is not None:
        ax.plot(t, energies, label=name)
        ax.legend(loc="upper right")
    else:
        ax.plot(t, energies)


def plotVelo(velo, ax, fig, name="", equal=None):
    ax.plot(np.arange(len(velo)), velo, "-", label=name)
    ax.legend(loc="upper right")
    if equal:
        ax.axis("equal")


def plotTraj(sol: np.ndarray, ax, fig, ani=True):
    x = sol[:, 0]
    y = sol[:, 1]
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.plot(x, y, linewidth=1)
    if ani:
        (line,) = ax.plot(x, y, color="k", linewidth=1)
        (point,) = ax.plot(x, y, "r*", markersize=10)
        return x, y, line, point


def plotPolarTraj(sol, ax, fig):
    r = sol[:, 0]
    theta = sol[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.plot(x, y)
    (line,) = ax.plot(x, y, color="k", linewidth=1)
    (point,) = ax.plot(x, y, "r*", markersize=10)
    return x, y, line, point


def plotMultipleTraj(sols: list, ax, fig, pos: int):
    """ pos is the position of arrows related to time length"""
    for sol in sols:
        x = sol[:, 0]
        y = sol[:, 1]
        ax.set_xlim([-4.5, 4.5])
        ax.set_ylim([-4.5, 4.5])
        ax.plot([x[0]], [y[0]], "ko")
        ax.plot(x, y)
        ax.annotate('',
                    xy=(x[pos + 2], y[pos + 2]), xycoords='data',
                    xytext=(x[pos], y[pos]), textcoords='data',
                    arrowprops=dict(arrowstyle="fancy, head_width=0.5, head_length=0.9, tail_width=0.5",
                                    connectionstyle="arc3", facecolor='black', alpha=0.4),
                    )
