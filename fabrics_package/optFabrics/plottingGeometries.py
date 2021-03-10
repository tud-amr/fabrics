import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plotMultipleTraj(sols, ax):
    for sol in sols:
        x = sol[:, 0]
        y = sol[:, 1]
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.plot(x, y)

def plotTraj(sols, axs):
    lines = []
    points = []
    for i, sol in enumerate(sols):
        axs[i].set_xlim([-5, 5])
        axs[i].set_ylim([-5, 5])
        x = sol[:, 0]
        y = sol[:, 1]
        axs[i].plot(x, y)
        line = axs[i].plot(x, y, color="k")[0]
        point = axs[i].plot(x, y, "r.")[0]
        lines.append(line)
        points.append(point)
    return lines, points

def animate(num, sols, lines, points):
    for i in range(len(lines)):
        maxNum = min(len(sols[i][:, 0])-1, num)
        start = max(0, num - 100)
        lines[i].set_data(sols[i][start:maxNum, 0], sols[i][start:maxNum, 1])
        points[i].set_data(sols[i][maxNum, 0], sols[i][maxNum, 1])
    return lines+ points

def plotMulti(sols, aniSols, fig, ax):
    ax = [a for b in ax for a in b]
    fig.suptitle("Optimization Fabrics")
    for i, sol in enumerate(sols):
        plotMultipleTraj(sol, ax[i])
    n = len(aniSols[0])
    (lines, points) = plotTraj(aniSols, ax)
    ani = animation.FuncAnimation(
        fig, animate, n,
        fargs=[aniSols, lines, points],
        interval=10, blit=True
    )
    plt.show()

def plot(sols):
    n = len(sols)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = [a for b in ax for a in b]
    fig.suptitle("Optimization Fabrics")
    (lines, points) = plotTraj(sols, ax[:n])
    ani = animation.FuncAnimation(
        fig, animate, 1000,
        fargs=[sols, lines, points],
        interval=10, blit=True
    )
    plt.show()
