import numpy as np
import cv2
import sys

try:
    import torch
except:
    pass

from constants import *

def segment_image(image, color):
    h, w, c = image.shape
    image = image.reshape((-1, c))
    dist_color = np.linalg.norm(image - color, axis=-1)
    dist_green = np.linalg.norm(image - GREEN, axis=-1)
    dist = np.vstack((dist_green, dist_color))
    return np.argmin(dist, axis=0).reshape((h, w))


def get_seg_idxs(image, color):
    seg = segment_image(image, color)
    locations = np.argwhere(seg).astype('float32')
    return locations


def preprocess_image(image, resize=True, to_torch=False):
    image = image[IMAGE_ORIGIN[0]:IMAGE_ORIGIN[0] + IMAGE_SIZE,
            IMAGE_ORIGIN[1]:IMAGE_ORIGIN[1] + IMAGE_SIZE, :]
    if resize:
        image = cv2.resize(image, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
    if to_torch and 'torch' in sys.modules:
        image = 2 * torch.FloatTensor(image.astype('float32') / 255.).permute(2, 0, 1).cuda() - 1
    return image


def coord_image_to_robot(image_coord):
    image_coord += 0.5
    image_coord *= float(IMAGE_SIZE) / IMAGE_INPUT_SIZE
    image_coord += IMAGE_ORIGIN.astype('float32')
    return image_coord.dot(A) + b


def rgb2hsv(r, g, b):
    assert 0 <= r < 256
    assert 0 <= g < 256
    assert 0 <= b < 256
    rp, gp, bp = r / 255., g / 255., b / 255.
    cmax, cmin = max(rp, gp, bp), min(rp, gp, bp)
    delta = cmax - cmin
    if delta == 0:
        h = 0
    elif cmax == rp:
        h = 60 * (((gp - bp) / delta) % 6)
    elif cmax == gp:
        h = 60 * (((bp - rp) / delta) + 2)
    elif cmax == bp:
        h = 60 * (((rp - gp) / delta) + 4)

    if cmax == 0:
        s = 0
    else:
        s = delta / cmax

    v = cmax
    return h, s, v


def hsv2rgb(h, s, v):
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if 0 <= h < 60:
        rp, gp, bp = (c, x, 0)
    elif 60 <= h < 120:
        rp, gp, bp = (x, c, 0)
    elif 120 <= h < 180:
        rp, gp, bp = (0, c, x)
    elif 180 <= h < 240:
        rp, gp, bp = (0, x, c)
    elif 240 <= h < 300:
        rp, gp, bp = (x, 0, c)
    elif 300 <= h < 360:
        rp, gp, bp = (c, 0, x)
    else:
        raise Exception()

    r, g, b = ((rp + m) * 255, (gp + m) * 255, (bp + m) * 255)
    return int(r), int(g), int(b)


def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0, gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)
