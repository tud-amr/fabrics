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
