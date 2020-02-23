import rospy
import moveit_commander
import geometry_msgs.msg
from pr2_controllers_msgs.msg import Pr2GripperCommand
from sensor_msgs.msg import Image
import tf.transformations as transform
import cv2

import cv_bridge
import pickle
import zmq
import zlib
import sys
import signal
import time

from constants import *


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
    arms = moveit_commander.MoveGroupCommander('arms')

    return left_gripper, right_gripper, left_arm, right_arm, arms

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
    pub.publish(Pr2GripperCommand(-0.01, 32))


def gripper_down(side):
    arm = get_arm(side)
    current_pose = arm.get_current_pose().pose

    pose_goal = geometry_msgs.msg.Pose()
    # pose_goal.position.x = current_pose.position.x
    # pose_goal.position.y = current_pose.position.y
    # pose_goal.position.z = current_pose.position.z
    pose_goal.position = current_pose.position
    pose_goal.orientation = geometry_msgs.msg.Quaternion(*GRIPPER_ORIENT_VERT)
    # pose_goal.orientation.x = 0.137595297951
    # pose_goal.orientation.y = 0.687421561519
    # pose_goal.orientation.z = -0.157305941758
    # pose_goal.orientation.w = 0.695538619653

    arm.set_pose_target(pose_goal)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()

    time.sleep(0.5)


def reset_arm(side):
    origin = ROBOT_ORIGIN[:]
    if side == 'right':
        origin[1] = -origin[1]
    move_arm(side, origin, GRIPPER_ORIENT_VERT)
    # gripper_down(side)


def reset_both_arms():
    left_origin = ROBOT_ORIGIN[:]
    right_origin = ROBOT_ORIGIN[:]
    right_origin[1] = -right_origin[1]
    move_both_arms(left_origin, right_origin, GRIPPER_ORIENT_HORI, GRIPPER_ORIENT_HORI)
    # orient = [0.137595297951, 0.687421561519, -0.157305941758, 0.695538619653]
    # gripper_down('left')
    # gripper_down('right')


def move_both_arms(left_loc, right_loc, left_orient=None, right_orient=None):
    for effector, loc, orient in [('l_wrist_roll_link', left_loc, left_orient), ('r_wrist_roll_link', right_loc, right_orient)]:
        old_pose = arms.get_current_pose(effector).pose
        x,y,z = loc
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z
        if orient:
            pose_goal.orientation = geometry_msgs.msg.Quaternion(*orient)
        else:
            pose_goal.orientation = old_pose.orientation
            # pose_goal.orientation.x = old_pose.orientation.x
            # pose_goal.orientation.y = old_pose.orientation.y
            # pose_goal.orientation.z = old_pose.orientation.z
            # pose_goal.orientation.w = old_pose.orientation.w
        arms.set_pose_target(pose_goal, effector)

    success = arms.go()
    arms.clear_pose_targets()

    for effector, loc in [('l_wrist_roll_link', left_loc), ('r_wrist_roll_link', right_loc)]:
        current_pose = arms.get_current_pose(effector).pose
        x,y,z = loc
        x_err = np.abs(x - current_pose.position.x)
        y_err = np.abs(y - current_pose.position.y)
        z_err = np.abs(z - current_pose.position.z)
        total_err = np.sqrt(x_err ** 2 + y_err ** 2 + z_err ** 2)
        print('py2::{} error: {}'.format(effector, total_err))

    if not success:
        print('py2::Two hand action failed...')
    return success


def move_arm(side, loc, orient=None):
    arm = get_arm(side)
    old_pose = arm.get_current_pose().pose
    x, y, z = loc
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = x
    pose_goal.position.y = y
    pose_goal.position.z = z
    if orient:
        pose_goal.orientation = geometry_msgs.msg.Quaternion(*orient)
    else:
        pose_goal.orientation = old_pose.orientation
        # pose_goal.orientation.x = old_pose.orientation.x
        # pose_goal.orientation.y = old_pose.orientation.y
        # pose_goal.orientation.z = old_pose.orientation.z
        # pose_goal.orientation.w = old_pose.orientation.w

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
        print('py2::{} arm action failed...'.format(side))
    return success


def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)


def coord_image_to_robot(image_coord):
    image_coord += 0.5
    image_coord *= float(IMAGE_SIZE) / IMAGE_INPUT_SIZE
    image_coord += IMAGE_ORIGIN.astype('float32')
    return image_coord.dot(A) + b


def signal_handler(signal, frame):
    print('\npy2::Ending process')
    sys.exit(0)



signal.signal(signal.SIGINT, signal_handler)

# global left_gripper, right_gripper, left_arm, right_arm, arms, send_message, waiting_to_send_image
left_gripper, right_gripper, left_arm, right_arm, arms = init_controller()


