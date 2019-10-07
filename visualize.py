import numpy as np
import matplotlib.pyplot as plt
import sys

W = np.load('weights.npy')
A, b = W[1:], W[0]

IMAGE_ORIGIN = np.array([160, 190], dtype='int32')
IMAGE_SIZE = 242
IMAGE_INPUT_SIZE = 64

def coord_image_to_robot(image_coord):
    # image_coord += 0.5
    # image_coord *= float(IMAGE_SIZE) / IMAGE_INPUT_SIZE
    # image_coord += IMAGE_ORIGIN.astype('float32')
    print(image_coord)
    return A.dot(image_coord) + b

# print(coord_image_to_robot(np.array([sys.argv[1], sys.argv[2]], dtype='float32')))
print(coord_image_to_robot([374, 338.5]))