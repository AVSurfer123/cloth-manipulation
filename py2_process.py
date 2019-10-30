import rospy
import moveit_commander
import geometry_msgs.msg
from pr2_controllers_msgs.msg import Pr2GripperCommand
from sensor_msgs.msg import Image

import cv_bridge
import pickle
import zmq
import zlib
import sys
import signal
import time

from constants import *


send_message = True
waiting_to_send_image = True
current_image = None

def init_socket():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://{}:{}".format(HOST, PORT))
    return socket


def init_controller():
    left_gripper = rospy.Publisher(LEFT_GRIPPER_TOPIC, Pr2GripperCommand, queue_size=10)
    right_gripper = rospy.Publisher(RIGHT_GRIPPER_TOPIC, Pr2GripperCommand, queue_size=10)
    #moveit_commander.roscpp_initialize(sys.argv)
    moveit_commander.roscpp_initialize(['joint_states:=/joint_states'])
    robot = moveit_commander.RobotCommander()
    left_arm = moveit_commander.MoveGroupCommander('left_arm')
    right_arm = moveit_commander.MoveGroupCommander('right_arm')

    return left_gripper, right_gripper, left_arm, right_arm

def get_arm(side):
    if side == 'left':
        return left_arm
    elif side == 'right':
        return right_arm
    else:
        return None

def get_gripper(side):
    if side == 'left':
        return left_gripper
    elif side == 'right':
        return right_gripper
    else:
        return None

def open_gripper(side):
    pub = get_gripper(side)
    pub.publish(Pr2GripperCommand(0.02, 32))


def close_gripper(side):
    pub = get_gripper(side)
    pub.publish(Pr2GripperCommand(0.0, 32))


def gripper_down(side):
    arm = get_arm(side)
    current_pose = arm.get_current_pose().pose

    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = current_pose.position.x
    pose_goal.position.y = current_pose.position.y
    pose_goal.position.z = current_pose.position.z
    pose_goal.orientation.x = 0.137595297951
    pose_goal.orientation.y = 0.687421561519
    pose_goal.orientation.z = -0.157305941758
    pose_goal.orientation.w = 0.695538619653

    arm.set_pose_target(pose_goal)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()

    time.sleep(0.5)


def reset_arm(side):
    origin = ROBOT_ORIGIN[:]
    if side == 'right':
        origin[1] = -origin[1]
    move_arm(side, *origin)
    gripper_down(side)


def move_both_arms(left_loc, right_loc):
    left_arm = get_arm('left')
    right_arm = get_arm('right')
    for arm, loc in [(left_arm, left_loc), (right_arm, right_loc)]:
        old_pose = arm.get_current_pose().pose
        x,y,z = loc
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z
        pose_goal.orientation.x = old_pose.orientation.x
        pose_goal.orientation.y = old_pose.orientation.y
        pose_goal.orientation.z = old_pose.orientation.z
        pose_goal.orientation.w = old_pose.orientation.w
        arm.set_pose_target(pose_goal)

    l_success = left_arm.go(wait=False)
    r_success = right_arm.go(wait=True)
    right_arm.stop()
    left_arm.stop()
    left_arm.clear_pose_targets()
    right_arm.clear_pose_targets()

    SIDES = ['left', 'right']
    for side, loc in zip(SIDES, (left_loc, right_loc)):
        arm = get_arm(side)
        current_pose = arm.get_current_pose().pose
        x,y,z = loc
        x_err = np.abs(x - current_pose.position.x)
        y_err = np.abs(y - current_pose.position.y)
        z_err = np.abs(z - current_pose.position.z)
        total_err = np.sqrt(x_err ** 2 + y_err ** 2 + z_err ** 2)
        print('py2::{} error: {}'.format(side, total_err))

    if not l_success:
        print('py2::Left action failed...')
    if not r_success:
        print('py2::Right action failed...')

def move_arm(side, x, y, z):
    arm = get_arm(side)
    old_pose = arm.get_current_pose().pose

    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = x
    pose_goal.position.y = y
    pose_goal.position.z = z
    pose_goal.orientation.x = old_pose.orientation.x
    pose_goal.orientation.y = old_pose.orientation.y
    pose_goal.orientation.z = old_pose.orientation.z
    pose_goal.orientation.w = old_pose.orientation.w

    arm.set_pose_target(pose_goal)
    success = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()

    current_pose = arm.get_current_pose().pose
    x_err = np.abs(pose_goal.position.x - current_pose.position.x)
    y_err = np.abs(pose_goal.position.y - current_pose.position.y)
    z_err = np.abs(pose_goal.position.z - current_pose.position.z)
    total_err = np.sqrt(x_err ** 2 + y_err ** 2 + z_err ** 2)

    print('py2::Error: {}'.format(total_err))
    if not success:
        print('py2::Action failed...')


def image_callback(msg):
    global send_message
    global waiting_to_send_image


    if send_message:
        send_message = False
        bridge = cv_bridge.CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        data = pickle.dumps(image)
        data = zlib.compress(data)
        socket.send(data)
        print('py2::Image sent')
        waiting_to_send_image = False


def execute_action(action, two_hand):
    picks, deltas = action
    if two_hand:
        left_loc, left_delta, right_loc, right_delta = picks[0], deltas[0], picks[1], deltas[1]

        left_start, left_end = left_loc, left_loc + left_delta * MAX_IMAGE_DELTA
        right_start, right_end = right_loc, right_loc + right_delta * MAX_IMAGE_DELTA
        left_start, left_end, right_start, right_end = map(coord_image_to_robot, [left_start, left_end, right_start, right_end])

        print('pr2::Moving left from pose {} to {}...'.format(left_start, left_end))
        print('pr2::Moving right from pose {} to {}...'.format(right_start, right_end))


        move_both_arms(left_start, right_start, Z_STATIONARY)
        time.sleep(0.5)
        move_both_arms(left_start, right_start, Z_DOWN)
        time.sleep(0.5)
        close_gripper(main_side)
        time.sleep(2.5)  # wait longer since close doesn't block
        move_arm(main_side, start_loc[0], start_loc[1], Z_UP)
        time.sleep(0.5)
        move_arm(main_side, end_loc[0], end_loc[1], Z_UP)
        time.sleep(0.5)
        move_arm(main_side, end_loc[0], end_loc[1], Z_DOWN)
        time.sleep(0.5)
        open_gripper(main_side)
        time.sleep(2.5)  # wait longer since open doesn't block
        move_arm(main_side, end_loc[0], end_loc[1], Z_UP)
        time.sleep(0.5)

        reset_arm(main_side)
        time.sleep(1)

    else:
        location, delta = picks[0], deltas[0]
        start_loc, end_loc = location, location + delta * MAX_IMAGE_DELTA
        start_loc = coord_image_to_robot(start_loc)
        end_loc = coord_image_to_robot(end_loc)

        print('pr2::Moving from pose {} to {}...'.format(start_loc, end_loc))

        move_arm(main_side, start_loc[0], start_loc[1], Z_STATIONARY)
        time.sleep(0.5)
        move_arm(main_side, start_loc[0], start_loc[1], Z_DOWN)
        time.sleep(0.5)
        close_gripper(main_side)
        time.sleep(2.5) # wait longer since close doesn't block
        move_arm(main_side, start_loc[0], start_loc[1], Z_UP)
        time.sleep(0.5)
        move_arm(main_side, end_loc[0], end_loc[1], Z_UP)
        time.sleep(0.5)
        move_arm(main_side, end_loc[0], end_loc[1], Z_DOWN)
        time.sleep(0.5)
        open_gripper(main_side)
        time.sleep(2.5) # wait longer since open doesn't block
        move_arm(main_side, end_loc[0], end_loc[1], Z_UP)
        time.sleep(0.5)

        reset_arm(main_side)
        time.sleep(1)
    print('pr2::Completed action')


def coord_image_to_robot(image_coord):
    image_coord += 0.5
    image_coord *= float(IMAGE_SIZE) / IMAGE_INPUT_SIZE
    image_coord += IMAGE_ORIGIN.astype('float32')
    return image_coord.dot(A) + b

def signal_handler(signal, frame):
    print('\npy2::Ending process')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    socket = init_socket()
    left_gripper, right_gripper, left_arm, right_arm = init_controller()
    main_side = 'right'  # Choose the side to use if one-handed

    rospy.init_node('towel_folding_py2')
    rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback)

    print('py2::Starting...')
    time_step = 0
    open_gripper()
    reset_arm()
    time.sleep(3)
    while time_step < N_ACTIONS:
        if waiting_to_send_image:
            continue

        data = socket.recv()
        data = zlib.decompress(data)
        action, two_hand = pickle.loads(data)

        print('pr2::Timestep {}'.format(time_step))
        print('pr2::Received action {}'.format(action))
        if two_hand:
             = action
        else:
            location, delta = action[:2], action[2:]
            assert len(location) == len(delta) == 2

        execute_action(action, two_hand)
        send_message = True
        waiting_to_send_image = True
        time_step += 1

