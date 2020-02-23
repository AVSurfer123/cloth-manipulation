import sys
sys.path.insert(1, '/home/owen/anaconda2/envs/softlearning/lib/python3.6/site-packages/cv2/')

import zmq
import pickle
import zlib

import math
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from softlearning.policies.utils import get_policy_from_variant
from softlearning.environments.utils import get_environment_from_params
from softlearning.value_functions.utils import get_Q_function_from_variant
import cv2
from datetime import datetime

from constants import *
from vision_utils import *

COLOR = WHITE

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


def update_image(image, picks, deltas):
    image, picks, deltas = image.copy(), picks.copy(), deltas.copy()
    image = preprocess_image(image, resize=False)

    # Image to label pick (yellow) and place (red) positions
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

        start_goal_image = image.copy()
        h, w = start_goal_image.shape[:2]
        start_goal_image[max(0, sr - radius):min(h, sr + radius), max(0, sc - radius):min(w, sc + radius)] = [255, 255, 0]
        start_goal_image[max(0, er - radius):min(h, er + radius), max(0, ec - radius):min(w, ec + radius)] = [255, 0, 0]

    ims[0].set_data(start_goal_image)

    if MODE != 'model_pick' and MODE != 'random_no_segmentation':
        # Image showing actions of perturbation positions
        image_input = cv2.resize(image, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
        locations = get_seg_idxs(image_input, COLOR)
        image_input = np.tile(image_input[None, :, :, :], (locations.shape[0], 1, 1, 1))
        tiled_locations = np.tile(locations, 50)
        scaled_locations = (locations + 0.5) * (float(IMAGE_SIZE) / IMAGE_INPUT_SIZE)
        all_actions = policy.actions_np([tiled_locations, image_input])[1]

        max_length = int(IMAGE_SIZE * 0.1)
        scale = 10
        location_image = cv2.resize(image.copy(), (IMAGE_SIZE * scale, IMAGE_SIZE * scale))

        for i in range(len(scaled_locations)):
            if i % 4 != 0:
                continue

            loc = (scaled_locations[i] * scale).astype('int32')
            act = all_actions[i, :2]
            act[1] = -act[1]
            act = act[[1, 0]]
            act *= max_length * scale

            startr, startc = loc
            endr, endc = loc + act
            endr, endc = int(endr), int(endc)

            cv2.arrowedLine(location_image, (startc, startr), (endc, endr), (255, 255, 255), 2 * scale)
        location_image = cv2.resize(location_image, (IMAGE_SIZE, IMAGE_SIZE))

        all_qs = [Q.predict([all_actions, tiled_locations, image_input]) for Q in Qs]
        all_qs = np.min(all_qs, axis=0)
        qmin, qmax = all_qs.min(), all_qs.max()
        print('Q-Val Min/Max: {}/{}'.format(qmin, qmax))

        start_color = rgb2hsv(0, 0, 255)
        end_color = rgb2hsv(255, 0, 0)

        # qval_image = image.copy()
        radius = math.ceil(float(IMAGE_SIZE) / IMAGE_INPUT_SIZE / 2)
        for i in range(len(scaled_locations)):
            loc = scaled_locations[i].astype('int32')
            qval = all_qs[i]
            alpha = (qval - qmin) / (qmax - qmin)
            color = [start_color[j] + alpha * (end_color[j] - start_color[j])
                     for j in range(3)]
            color = hsv2rgb(*color)
            location_image[loc[0] - radius:loc[0] + radius, loc[1] - radius:loc[1] + radius] = color

        ims[1].set_data(location_image)

        # Image showing heat map of Q-values over locations on the cloth

        # ims[2].set_data(location_image)

    # Image showing segmentation
    seg_image = cv2.resize(image.copy(), (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))

    idxs = get_seg_idxs(seg_image, COLOR).astype('int32')
    seg_image[:] = 0
    for r, c in idxs:
        seg_image[r, c, :] = 255

    rtn_image = seg_image.copy()
    binary_image = seg_image.copy()
    binary_image = (binary_image == 255).all(axis=-1).astype(int)
    line = np.linspace(0, 31, num=32) * -1
    column = np.concatenate([np.flip(line), line])
    reward = np.sum(binary_image * np.exp(column).reshape((64, 1))) / 111.0

    seg_image = cv2.resize(seg_image, (IMAGE_SIZE, IMAGE_SIZE))
    ims[3].set_data(seg_image)

    # Image showing down-sampled version
    downsampled_image = cv2.resize(image.copy(), (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
    ims[4].set_data(downsampled_image)

    # Update all plots
    fig.canvas.draw()

    plt.savefig(os.path.join(folder, 'observations', '{}.png'.format(time_step)))

    return reward, rtn_image


def generate_action(policy, image, mode):
    image = preprocess_image(image)
    locations = get_seg_idxs(image, COLOR)
    two_hand = False
    if mode == 'two_hand_maxq':
        two_hand = True
        pass
    elif mode == 'two_hand_policy_spread':
        two_hand = True
        pass
    elif mode == 'two_hand_random_spread':
        two_hand = True
        pass
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
        idx = idxs[np.argmax(np.log(all_qs) + gumbel)]

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
        picks = np.expand_dims(location, axis=0)
        deltas = np.expand_dims(delta, axis=0)

    print("Picks:", picks)
    print("Deltas:", deltas)

    # Convert from Cartesian x,y delta to image x,y delta
    for i in range(len(deltas)):
        deltas[i][1] = -deltas[i][1]
        deltas[i] = deltas[i][[1, 0]]

    # Point list is length 2 if two_hand else 1
    # RETURN TYPE: ((pick locations: List[Tuple[float, float]], deltas: List[Tuple[float, float]]), two_hand: bool)
    return picks.astype('float32'), deltas.astype('float32'), two_hand


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
    r1 = []
    while True:
        print('py3::Waiting for image...')
        data = socket.recv()
        data = zlib.decompress(data)
        image = pickle.loads(data, encoding='latin1')
        print('py3::Received image, executing policy')
        picks, deltas, two_hand = generate_action(policy, image, mode=MODE)
        reward, binary_image = update_image(image, picks, deltas)
        print('Reward: {:.4f}'.format(reward))

        r1.append(reward)

        cv2.imwrite(os.path.join(folder, 'full_observations', '{}.png'.format(time_step)),
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(folder, 'raw_observations', '{}.png'.format(time_step)),
                    cv2.cvtColor(preprocess_image(image, resize=False), cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(folder, 'segmentations', '{}.png'.format(time_step)), binary_image)

        np.save(os.path.join(folder, 'rewards', 'intersection.npy'), r1)

        print('py3::Sending action')
        data = pickle.dumps((picks, deltas, two_hand), protocol=2)
        data = zlib.compress(data)
        socket.send(data)
        time_step += 1