import sys
sys.path.insert(1, '/home/owen/anaconda2/envs/softlearning/lib/python3.6/site-packages/cv2/')

import zmq
import pickle
import zlib

import os
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import torch

from constants import *


def init_socket():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:{}".format(PORT))
    return socket


def init_policy(path):
    checkpoint = torch.load(path, map_location='cuda')
    encoder, trans = checkpoint['encoder'], checkpoint['trans']
    return encoder, trans


def coord_image_to_robot(image_coord):
    image_coord += 0.5
    image_coord *= float(IMAGE_SIZE) / IMAGE_INPUT_SIZE
    image_coord += IMAGE_ORIGIN.astype('float32')
    return image_coord.dot(A) + b


def update_image(image, action):
    image, action = image.copy(), action.copy()
    location, delta = action[:2], action[2:]
    image = preprocess_image(image, resize=False)

    # Image to label pick (yellow) and place (red) positions
    start_loc, end_loc = location, location + delta * MAX_IMAGE_DELTA
    start_loc, end_loc = coord_image_to_robot(start_loc), coord_image_to_robot(end_loc)
    start_loc = (start_loc - b).dot(np.linalg.inv(A))
    end_loc = (end_loc - b).dot(np.linalg.inv(A))
    start_loc, end_loc = start_loc - IMAGE_ORIGIN, end_loc - IMAGE_ORIGIN

    start_loc, end_loc = start_loc.astype('int32'), end_loc.astype('int32')
    sr, sc = start_loc
    er, ec = end_loc
    radius = 4

    start_goal_image = image.copy()
    h, w = start_goal_image.shape[:2]
    start_goal_image[max(0, sr - radius):min(h, sr + radius), max(0, sc - radius):min(w, sc + radius)] = [255, 255, 0]
    start_goal_image[max(0, er - radius):min(h, er + radius), max(0, ec - radius):min(w, ec + radius)] = [255, 0, 0]

    ims[0].set_data(start_goal_image)

    # Image showing segmentation
    seg_image = cv2.resize(image.copy(), (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))

    idxs = get_seg_idxs(seg_image).astype('int32')
    seg_image[:] = 0
    for r, c in idxs:
        seg_image[r, c, :] = 255
    seg_image = cv2.resize(seg_image, (IMAGE_SIZE, IMAGE_SIZE))
    ims[1].set_data(seg_image)

    # Image showing down-sampled version
    downsampled_image = cv2.resize(image.copy(), (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
    ims[2].set_data(downsampled_image)

    # Update all plots
    fig.canvas.draw()

    plt.savefig(os.path.join(folder, 'observations', '{}.png'.format(time_step)))

    return seg_image


def segment_image(image):
    h, w, c = image.shape
    image = image.reshape((-1, c))
    dist_white = np.linalg.norm(image - WHITE, axis=-1)
    dist_green = np.linalg.norm(image - GREEN, axis=-1)
    dist = np.vstack((dist_green, dist_white))
    return np.argmin(dist, axis=0).reshape((h, w))


def get_seg_idxs(image):
    seg = segment_image(image)
    locations = np.argwhere(seg).astype('float32')
    return locations


def run_single(model, *args):
    return model(*[a.unsqueeze(0) for a in args]).squeeze(0)


def sample_actions(locations, n):
    locs = locations[np.random.randint(len(locations), size=(n,))]
    deltas = 2 * np.random.rand(n, 2) - 1
    return np.concatenate((locs, deltas), axis=1)


def generate_action(encoder, trans, current_image, goal_image):
    locations = 2 * (get_seg_idxs(preprocess_image(current_image)) / 63.) - 1
    current_image = preprocess_image(current_image, to_torch=True)

    z_current, z_goal = run_single(encoder, current_image), run_single(encoder, goal_image)
    z_current, z_goal = z_current.unsqueeze(0), z_goal.unsqueeze(0)
    n_trials = 1000
    with torch.no_grad():
        actions = torch.FloatTensor(sample_actions(locations, n_trials)).cuda()
        zs = trans(z_current.repeat(n_trials, 1), actions)
        dists = torch.norm((zs - z_goal).view(n_trials, -1), dim=-1)
        idx = torch.argmin(dists)
    action = actions[idx].cpu().numpy()
    action[:2] = (action[:2] * 0.5 + 0.5) * 63
    return action


def preprocess_image(image, resize=True, to_torch=False):
    image = image[IMAGE_ORIGIN[0]:IMAGE_ORIGIN[0] + IMAGE_SIZE,
            IMAGE_ORIGIN[1]:IMAGE_ORIGIN[1] + IMAGE_SIZE, :]
    if resize:
        image = cv2.resize(image, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
    if to_torch:
        image = 2 * torch.FloatTensor(image.astype('float32') / 255.).permute(2, 0, 1).cuda() - 1
    return image


if __name__ == '__main__':
    name = datetime.now().isoformat() + '_{}'.format(sys.argv[1])
    folder = os.path.join('images', EXPERIMENT_NAME, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.makedirs(os.path.join(folder, 'full_observations'))
    os.makedirs(os.path.join(folder, 'raw_observations'))
    os.makedirs(os.path.join(folder, 'observations'))
    os.makedirs(os.path.join(folder, 'segmentations'))
    os.makedirs(os.path.join(folder, 'rewards'))

    socket = init_socket()
    encoder, trans = init_policy(POLICY_PATH)
    goal_image = cv2.resize(cv2.cvtColor(cv2.imread(GOAL_IMAGE), cv2.COLOR_BGR2RGB), (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))

    dummy_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ims = []
    ims.append(axs[0, 0].imshow(dummy_img.copy()))
    ims.append(axs[1, 0].imshow(dummy_img.copy()))
    ims.append(axs[0, 1].imshow(dummy_img.copy()))
    ims.append(axs[1, 1].imshow(goal_image))
    fig.canvas.draw()

    goal_image = 2 * torch.FloatTensor(goal_image.astype('float32') / 255.).permute(2, 0, 1).cuda() - 1


    print('py3::Starting...')
    time_step = 0
    r1 = []
    while True:
        print('py3::Waiting for image...')
        data = socket.recv()
        data = zlib.decompress(data)
        image = pickle.loads(data, encoding='latin1')
        print('py3::Received image, executing policy')

        action = generate_action(encoder, trans, image, goal_image).astype('double')

        binary_image = update_image(image, action)
        cv2.imwrite(os.path.join(folder, 'full_observations', '{}.png'.format(time_step)),
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(folder, 'raw_observations', '{}.png'.format(time_step)),
                    cv2.cvtColor(preprocess_image(image, resize=False), cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(folder, 'segmentations', '{}.png'.format(time_step)), binary_image)

        np.save(os.path.join(folder, 'rewards', 'intersection.npy'), r1)

        print('py3::Sending action')
        data = pickle.dumps(action, protocol=2)
        data = zlib.compress(data)
        socket.send(data)
        time_step += 1