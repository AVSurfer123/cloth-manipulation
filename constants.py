import numpy as np

W = np.load('weights.npy')
A = W[1:]
b = W[0]

Z_UP = 1.075
Z_DOWN = 1.02
Z_STATIONARY = 1.2
RIGHT_Z_OFFSET = -.04

# ROBOT_ORIGIN = [0.2, 0.55, Z_STATIONARY]
GRIPPER_ORIENT_HORI = [0.137595297951, 0.687421561519, -0.157305941758, 0.695538619653]
GRIPPER_ORIENT_VERT = [-0.48673041075, 0.509701024424, 0.501773446261, 0.501519472782]
GRIPPER_OPEN = 0.03
LEFT_ROBOT_ORIGIN = [0.34, 0.7, Z_STATIONARY]
RIGHT_ROBOT_ORIGIN = [.34, -.5, Z_STATIONARY]

GOAL_IMAGE = 'images/3pi_over_4.png'
POLICY_PATH = 'policies/cpc_torch/rope_nce/checkpoint'
EXPERIMENT_NAME = 'rope_test'

RIGHT_GRIPPER_TOPIC = '/r_gripper_controller/command'
LEFT_GRIPPER_TOPIC = '/l_gripper_controller/command'

N_ACTIONS = 50

# CHECKPOINT_PATH = '/home/owen/wilson/cloth-manipulation/policies/tf/cloth_multiple/checkpoint_450'
# CHECKPOINT_PATH = '/home/owen/wilson/cloth-manipulation/policies/tf/rope_seed_9029/checkpoint_90'


# Both

HOST = 'localhost'
PORT = 7778
IMAGE_ORIGIN = np.array([80, 125], dtype='int32')
IMAGE_SIZE = 300
IMAGE_INPUT_SIZE = 64

MAX_IMAGE_DELTA = 10
GAMMA_CORRECTION = .5

BLUE = np.array([[0, 0, 255]])
GREEN = np.array([[0, 255, 0]])
WHITE = np.array([[255, 255, 255]])
YELLOW = np.array([[255, 255, 0]])

MODE = 'q_percentile'
TEMPERATURE = 1.0
PERCENTILE = 75
POLICY_NAME = 'cloth_multi_video'

CLOTH_WIDTH = int(38.3)
CLOTH_HEIGHT = int(36.4)


ROPE_GRIPPER_POS = (-0.505492181707, 0.492453067403, 0.478780032014, 0.522242579252)

TEST_GRIPPER_POS = (0.137595297951, 0.687421561519, -0.157305941758, 0.695538619653)
