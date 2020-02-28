import numpy as np


image_coords = np.array([
    [1,98,314],
    [1,77,158],
    [1,355,150],
    [1,401,312],
    [1,261,225],
])

coord = image_coords.copy()

coord[:, 1] = image_coords[:, 2]
coord[:, 2] = image_coords[:, 1]
image_coords = coord

robot_coords = np.array([
    [0.5,0.3],
    [0.8,0.3],
    [0.7,-0.2],
    [.4,-.2],
    [.6,.0]
])


weights = np.linalg.lstsq(image_coords, robot_coords)[0]
print(weights)
test = np.array([1,100,150])
print(image_coords.dot(weights))
print(robot_coords)
np.save('weights.npy', weights)
