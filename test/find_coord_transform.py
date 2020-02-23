import numpy as np


image_coords = np.array([
    [1,71,190],
    [1,13,310],
    [1,181,185],
    [1,5,228],
    [1,124,326],
])

coord = image_coords.copy()

coord[:, 1] = image_coords[:, 2]
coord[:, 2] = image_coords[:, 1]
image_coords = coord

robot_coords = np.array([
    [0.7,0.4],
    [0.5,0.5],
    [0.7,0.2],
    [.65,.52],
    [.45,.32]
])


weights = np.linalg.lstsq(image_coords, robot_coords)[0]
print(weights)
test = np.array([1,100,150])
print(image_coords.dot(weights))
print(robot_coords)
np.save('taped_weights.npy', weights)
