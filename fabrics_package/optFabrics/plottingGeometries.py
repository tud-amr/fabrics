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

def forwardKinematics(q, n):
    l = np.array([1.0, 1.0, 1.0])
    fk = np.array([0.0, 0.2, 0.0])
    for i in range(n):
        angle = 0.0
        for j in range(i+1):
            angle += q[j]
        fk[0] += np.cos(angle) * l[i]
        fk[1] += np.sin(angle) * l[i]
        fk[2] += q[i]
    return fk

def plotRobot(q, n, ax):
    joints = []
    links = []
    offset = np.array([-0.00, -0.05])
    xi = [np.array([0, 0.2, 0])]
    patches = []
    for i in range(n):
        xi.append(forwardKinematics(q, i+1))
        a = xi[i+1][2]
        R = np.array([
                        [np.cos(a), -np.sin(a)],
                        [np.sin(a), np.cos(a)]
                    ])
        offset_real = np.dot(R, offset)
        joints.append(plt.Circle(xi[i+1][0:2], radius=0.1))
        links.append(plt.Rectangle(xi[i][0:2] + offset_real, 1.0, 0.1, angle = np.rad2deg(xi[i+1][2])))
        patches.append(ax.add_patch(joints[i]))
        patches.append(ax.add_patch(links[i]))
    return patches

def animate2DRobot(num, qs, n, ax):
    q = qs[:, num]
    lenTrail = 50
    start = max(0, num - lenTrail)
    x_ee = np.zeros((lenTrail, 2))
    line = ax.plot(0, 0, color='r')[0]
    for j in range(lenTrail):
        index = min(len(qs[0]) - 1, start+j)
        x_ee[j] = forwardKinematics(qs[:, index], n)[0:2]
    line.set_data(x_ee[:, 0], x_ee[:, 1])
    return [line] + plotRobot(q, n, ax)

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
        interval=10, blit=True, save_count=10000
    )
    plt.show()
    print("Saving the figure can take several minutes")
    fileName = input("Enter filename or 'no'\n")
    if fileName != 'no':
        print("Saving the figure ...")
        writerVideo = animation.FFMpegWriter(fps=60)
        ani.save(fileName, writer=writerVideo)

def plot2DRobot(qs, fig, ax, dim):
    fig.suptitle("Optimization Fabrics")
    n = len(qs[0])
    x_ee = np.zeros((n, 2))
    for j in range(len(qs[0])):
        x_ee[j] = forwardKinematics(qs[:, j], dim)[0:2]
    ax.plot(x_ee[:, 0], x_ee[:, 1])
    ani = animation.FuncAnimation(
        fig, animate2DRobot, n, 
        fargs=[qs, dim, ax],
        interval=10, blit=True, repeat=True
    )
    plt.show()

def plot(sols, fig, ax):
    n = len(sols)
    ax = [a for b in ax for a in b]
    fig.suptitle("Optimization Fabrics")
    n = len(sols[0])
    (lines, points) = plotTraj(sols, ax[:n])
    ani = animation.FuncAnimation(
        fig, animate, n,
        fargs=[sols, lines, points],
        interval=10, blit=True
    )
    plt.show()

def plotObstacle(obst, axs):
    x = obst[0]
    r = obst[1]
    axs = [a for b in axs for a in b]
    for ax in axs:
        o = plt.Circle(x, r, color='r')
        ax.add_patch(o)

