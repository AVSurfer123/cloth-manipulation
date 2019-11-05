import sys
sys.path.insert(1, '/home/owen/anaconda2/envs/softlearning/lib/python3.6/site-packages/cv2/')

import zmq
import pickle
import zlib

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from softlearning.policies.utils import get_policy_from_variant
from softlearning.environments.utils import get_environment_from_params
from softlearning.value_functions.utils import get_Q_function_from_variant
import cv2
from datetime import datetime

from constants import *


def init_socket():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:{}".format(PORT))
    return socket


def init_policy():
    session = tf.keras.backend.get_session()
    checkpoint_path = CHECKPOINT_PATH.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.pkl')
    with open(variant_path, 'rb') as f:
        variant = pickle.load(f)

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = (
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])
    environment_params['n_parallel_envs'] = 1
    evaluation_environment = get_environment_from_params(environment_params)
    policy = get_policy_from_variant(variant, evaluation_environment)
    policy.set_weights(picklable['policy_weights'])

    Qs = get_Q_function_from_variant(variant, evaluation_environment)
    for i, Q in enumerate(Qs):
        Qs[i].load_weights(os.path.join(checkpoint_path, 'Qs_{}'.format(i)))

    return policy, Qs


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


def update_image(image, action):
    image, action = image.copy(), action.copy()
    picks, deltas = action
    image = preprocess_image(image, resize=False)

    # Image to label pick (yellow) and place (red) positions
    start_goal_image = image.copy()
    h, w = start_goal_image.shape[:2]
    for location, delta in zip(picks, deltas):
        start_loc, end_loc = location, location + delta * MAX_IMAGE_DELTA
        start_loc, end_loc = coord_image_to_robot(start_loc), coord_image_to_robot(end_loc)
        start_loc = (start_loc - b).dot(np.linalg.inv(A))
        end_loc = (end_loc - b).dot(np.linalg.inv(A))
        start_loc, end_loc = start_loc - IMAGE_ORIGIN, end_loc - IMAGE_ORIGIN

        start_loc, end_loc = start_loc.astype('int32'), end_loc.astype('int32')
        sr, sc = start_loc
        er, ec = end_loc
        radius = 4

        start_goal_image[max(0, sr-radius):min(h, sr+radius), max(0, sc-radius):min(w, sc+radius)] = [0, 0, 0]
        start_goal_image[max(0, er-radius):min(h, er+radius), max(0, ec-radius):min(w, ec+radius)] = [255, 0, 0]

    ims[0].set_data(start_goal_image)

    if MODE != 'model_pick' and MODE != 'random_no_segmentation':
        # Image showing actions of perturbation positions
        image_input = cv2.resize(image, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
        locations = get_seg_idxs(image_input)
        image_input = np.tile(image_input[None, :, :, :], (locations.shape[0], 1, 1, 1))
        tiled_locations = np.tile(locations, 50)
        scaled_locations = (locations + 0.5) * (float(IMAGE_SIZE) / IMAGE_INPUT_SIZE)
        all_actions = policy.actions_np([tiled_locations, image_input])[1]

        max_length = int(IMAGE_SIZE * 0.1)
        location_image = image.copy()
        for i in range(len(scaled_locations)):
            if i % 16 != 0:
                continue

            loc = scaled_locations[i]
            act = all_actions[i, :2]
            act[1] = -act[1]
            act = act[[1, 0]]
            act *= max_length

            startr, startc = loc
            endr, endc = loc + act
            endr, endc = int(endr), int(endc)

            cv2.arrowedLine(location_image, (startc, startr), (endc, endr), (255, 255, 255), 2)
        ims[1].set_data(location_image)

        # Image showing heat map of Q-values over locations on the cloth
        all_qs = [Q.predict([all_actions, tiled_locations, image_input]) for Q in Qs]
        all_qs = np.min(all_qs, axis=0)
        qmin, qmax = all_qs.min(), all_qs.max()
        print('Q-Val Min/Max: {}/{}'.format(qmin, qmax))

        start_color = rgb2hsv(0, 0, 255)
        end_color = rgb2hsv(255, 0, 0)
        qval_image = image.copy()
        radius = int(float(IMAGE_SIZE) / IMAGE_INPUT_SIZE / 2)
        for i in range(len(scaled_locations)):
            loc = scaled_locations[i].astype('int32')
            qval = all_qs[i]
            alpha = (qval - qmin) / (qmax - qmin)
            color = [start_color[j] + alpha * (end_color[j] - start_color[j])
                     for j in range(3)]
            color = hsv2rgb(*color)
            qval_image[loc[0]-radius:loc[0]+radius, loc[1]-radius:loc[1]+radius] = color
        ims[2].set_data(qval_image)

    # Image showing segmentation
    seg_image = cv2.resize(image.copy(), (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))

    ground_truth_seg = np.zeros((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE), dtype=np.uint8)
    start_row, start_col = int((IMAGE_INPUT_SIZE - CLOTH_HEIGHT) / 2), int((IMAGE_INPUT_SIZE - CLOTH_WIDTH) / 2)
    ground_truth_seg[start_row:start_row + CLOTH_HEIGHT, start_col:start_col + CLOTH_WIDTH] = 1

    idxs = get_seg_idxs(seg_image).astype('int32')
    seg_image[:] = 0
    for r, c in idxs:
        seg_image[r, c, :] = 255

    rtn_image = seg_image.copy()
    binary_image = seg_image.copy()
    binary_image = (binary_image == 255).all(axis=-1)
    reward_intersection = (binary_image & ground_truth_seg).sum() / ground_truth_seg.sum()
    reward_iou = (binary_image & ground_truth_seg).sum() / (binary_image | ground_truth_seg).sum(self.actions_model_for_fixed_latents)

    seg_image[start_row, start_col:start_col + CLOTH_WIDTH] = [255, 0, 0]
    seg_image[start_row + CLOTH_HEIGHT, start_col:start_col + CLOTH_WIDTH] = [255, 0, 0]
    seg_image[start_row:start_row + CLOTH_HEIGHT, start_col] = [255, 0, 0]
    seg_image[start_row:start_row + CLOTH_HEIGHT, start_col + CLOTH_WIDTH] = [255, 0, 0]

    seg_image = cv2.resize(seg_image, (IMAGE_SIZE, IMAGE_SIZE))
    ims[3].set_data(seg_image)

    # Image showing down-sampled version
    downsampled_image = cv2.resize(image.copy(), (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
    ims[4].set_data(downsampled_image)

    # Update all plots
    fig.canvas.draw()

    plt.savefig(os.path.join(folder, 'observations', '{}.png'.format(time_step)))

    return reward_intersection, reward_iou, rtn_image


def segment_image(image):
    h, w, c = image.shape
    image = image.reshape((-1, c))
    dist_blue = np.linalg.norm(image - BLUE, axis=-1)
    dist_green = np.linalg.norm(image - GREEN, axis=-1)
    dist = np.vstack((dist_green, dist_blue))
    return np.argmin(dist, axis=0).reshape((h, w))


def get_seg_idxs(image):
    seg = segment_image(image)
    locations = np.argwhere(seg).astype('float32')
    return locations


def generate_action(policy, image, mode):
    image = preprocess_image(image)
    locations = get_seg_idxs(image)
    sorted_loc = np.sort(locations, axis=0)
    left_loc = sorted_loc[sorted_loc[:, 0] <= IMAGE_INPUT_SIZE//2]
    right_loc = sorted_loc[sorted_loc[:, 0] > IMAGE_INPUT_SIZE//2]
    two_hand = False
    if mode == 'two_hand_maxq': # TODO
        image_input = np.tile(image[None, :, :, :], (locations.shape[0], 1, 1, 1))
        cv2.imshow(image_input)
        tiled_locations = np.tile(locations, 50)
        all_actions = policy.actions_np([tiled_locations, image_input])[1]
        all_qs = [Q.predict([all_actions, tiled_locations, image_input]) for Q in Qs]
        all_qs = np.min(all_qs, axis=0)

        threshold = np.percentile(all_qs, PERCENTILE)
        idxs = np.arange(len(all_qs))[:, None][all_qs > threshold]
        all_qs = all_qs[all_qs > threshold]
        all_qs = (all_qs - all_qs.min()) / (all_qs.max() - all_qs.min())
        all_qs /= TEMPERATURE
        all_qs -= all_qs.max()
        all_qs = np.exp(all_qs)
        all_qs /= all_qs.sum()

        uniform = np.random.rand(*all_qs.shape)
        uniform = np.clip(uniform, 1e-5, 1 - 1e-5)
        gumbel = -np.log(-np.log(uniform))
        idx = idxs[np.argmax(all_qs + gumbel)]

        print('Percentile', threshold)

        location = locations[idx]
        delta = all_actions[idx, :2]
        two_hand = True

    elif mode == 'two_hand_policy_spread': # TODO
        two_hand = True
    elif mode == 'two_hand_random_spread': # TODO
        left_pick = np.random.choice(left_loc.shape, 1)
        left_delta = np.random.uniform(-1, 0, (2, 1))
        dist = np.linalg.norm(right_loc - left_pick)
        right_pick = right_loc[np.argmax(dist)]
        right_delta = -left_delta
        picks = np.array([left_pick, right_pick])
        deltas = np.array([left_delta, right_delta])
        two_hand = True

    elif mode == 'maxq_sample':
        print('Using maxq_sample')
        image_input = np.tile(image[None, :, :, :], (locations.shape[0], 1, 1, 1))
        tiled_locations = np.tile(locations, 50)
        all_actions = policy.actions_np([tiled_locations, image_input])[1]
        all_qs = [Q.predict([all_actions, tiled_locations, image_input]) for Q in Qs]
        all_qs = np.min(all_qs, axis=0)
        all_qs = (all_qs - all_qs.min()) / (all_qs.max() - all_qs.min())

        all_qs /= TEMPERATURE
        all_qs -= all_qs.max()
        all_qs = np.exp(all_qs)
        all_qs /= all_qs.sum()

        uniform = np.random.rand(*all_qs.shape)
        uniform = np.clip(uniform, 1e-5, 1 - 1e-5)
        gumbel = -np.log(-np.log(uniform))
        idx = np.argmax(all_qs + gumbel)

        location = locations[idx]
        delta = all_actions[idx, :2]
    elif mode == 'q_percentile':
        print('Using q_percentile')
        image_input = np.tile(image[None, :, :, :], (locations.shape[0], 1, 1, 1))
        tiled_locations = np.tile(locations, 50)
        all_actions = policy.actions_np([tiled_locations, image_input])[1]
        all_qs = [Q.predict([all_actions, tiled_locations, image_input]) for Q in Qs]
        all_qs = np.min(all_qs, axis=0)

        threshold = np.percentile(all_qs, PERCENTILE)
        idxs = np.arange(len(all_qs))[:, None][all_qs > threshold]
        all_qs = all_qs[all_qs > threshold]
        all_qs = (all_qs - all_qs.min()) / (all_qs.max() - all_qs.min())
        all_qs /= TEMPERATURE
        all_qs -= all_qs.max()
        all_qs = np.exp(all_qs)
        all_qs /= all_qs.sum()

        uniform = np.random.rand(*all_qs.shape)
        uniform = np.clip(uniform, 1e-5, 1 - 1e-5)
        gumbel = -np.log(-np.log(uniform))
        idx = idxs[np.argmax(all_qs + gumbel)]

        print('Percentile', threshold)

        location = locations[idx]
        delta = all_actions[idx, :2]
    elif mode == 'maxq_nosample':
        print('Using maxq_nosample')
        image_input = np.tile(image[None, :, :, :], (locations.shape[0], 1, 1, 1))
        tiled_locations = np.tile(locations, 50)
        all_actions = policy.actions_np([tiled_locations, image_input])[1]
        all_qs = [Q.predict([all_actions, tiled_locations, image_input]) for Q in Qs]
        all_qs = np.min(all_qs, axis=0)

        idx = np.argmax(all_qs)

        location = locations[idx]
        delta = all_actions[idx, :2]
    elif mode == 'random_pick':
        print('random_pick')
        location = locations[np.random.randint(len(locations))]
        tiled_location = np.tile(location, 50)
        delta = policy.actions_np([tiled_location[None, :], image[None, :]])[1][0, :2]
    elif mode == 'random_pick_place':
        print('random_pick_place')
        location = locations[np.random.randint(len(locations))]
        delta = np.random.rand(2).astype('float32')
        delta = 2 * delta - 1
    elif mode == 'model_pick':
        print('model_pick')
        delta, location = policy.actions_np([image[None, :, :, :]])
        delta, location = delta[0], location[0]
        location = (location * 0.5 + 0.5) * 63
    elif mode == 'random_no_segmentation':
        print('random_no_segmentation')
        location = np.random.randint(0, 64, size=(2,))
        delta = np.random.rand(2) * 2 - 1
    else:
        raise Exception(mode)

    if not two_hand:
        picks = np.array([location])
        deltas = np.array([delta])

    # Convert from Cartesian x,y to image x,y
    for i in range(len(deltas)):
        deltas[i][1] = -deltas[i][1]
        deltas[i] = deltas[i][[1, 0]]

    # Point list is length 2 if two_hand else 1
    # RETURN TYPE: ((pick locations: List[Tuple[float, float]], deltas: List[Tuple[float, float]]), two_hand: bool)
    return (picks.astype('float32'), deltas.astype('float32')), two_hand


def preprocess_image(image, resize=True):
    image = image[IMAGE_ORIGIN[0]:IMAGE_ORIGIN[0] + IMAGE_SIZE,
            IMAGE_ORIGIN[1]:IMAGE_ORIGIN[1] + IMAGE_SIZE, :]
    if resize:
        image = cv2.resize(image, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
    return image


if __name__ == '__main__':
    name = datetime.now().isoformat() + '_{}'.format(sys.argv[1])
    folder = os.path.join('images', POLICY_NAME, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.makedirs(os.path.join(folder, 'full_observations'))
    os.makedirs(os.path.join(folder, 'raw_observations'))
    os.makedirs(os.path.join(folder, 'observations'))
    os.makedirs(os.path.join(folder, 'segmentations'))
    os.makedirs(os.path.join(folder, 'rewards'))

    socket = init_socket()
    policy, Qs = init_policy()

    dummy_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    plt.ion()
    fig, axs = plt.subplots(2, 3, figsize=(12, 9))
    ims = []
    ims.append(axs[0, 0].imshow(dummy_img.copy()))
    ims.append(axs[1, 0].imshow(dummy_img.copy()))
    ims.append(axs[0, 1].imshow(dummy_img.copy()))
    ims.append(axs[1, 1].imshow(dummy_img.copy()))
    ims.append(axs[0, 2].imshow(np.zeros((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3), dtype=np.uint8)))
    fig.canvas.draw()

    print('py3::Starting...')
    time_step = 0
    r1, r2 = [], []
    while True:
        print('py3::Waiting for image...')
        data = socket.recv()
        data = zlib.decompress(data)
        image = pickle.loads(data, encoding='latin1')
        print('py3::Received image, executing policy')
        action, two_hand = generate_action(policy, image, mode=MODE)
        reward_intersection, reward_iou, binary_image = update_image(image, action)
        print('Reward Intersection: {:.4f}, Reward IOU: {:.4f}'.format(reward_intersection, reward_iou))

        r1.append(reward_intersection)
        r2.append(reward_iou)

        cv2.imwrite(os.path.join(folder, 'full_observations', '{}.png'.format(time_step)),
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(folder, 'raw_observations', '{}.png'.format(time_step)),
                    cv2.cvtColor(preprocess_image(image, resize=False), cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(folder, 'segmentations', '{}.png'.format(time_step)), binary_image)

        np.save(os.path.join(folder, 'rewards', 'intersection.npy'), r1)
        np.save(os.path.join(folder, 'rewards', 'iou.npy'), r2)

        print('py3::Sending action')
        data = pickle.dumps((action, two_hand), protocol=2)
        data = zlib.compress(data)
        socket.send(data)
        time_step += 1