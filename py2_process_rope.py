import rospy
import moveit_commander
import geometry_msgs.msg
from pr2_controllers_msgs.msg import Pr2GripperCommand
from sensor_msgs.msg import Image

import cv2
import cv_bridge
import pickle
import zmq
import zlib
import sys
import os
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
    pub = rospy.Publisher(TOPIC_NAME, Pr2GripperCommand, queue_size=10)
    moveit_commander.roscpp_initialize(sys.argv)
    robot = moveit_commander.RobotCommander()
    left_arm = moveit_commander.MoveGroupCommander('left_arm')

    return pub, left_arm


def open_gripper():
    pub.publish(Pr2GripperCommand(0.04, 32))


def close_gripper():
    pub.publish(Pr2GripperCommand(-0.02, 32))


def gripper_down():
    current_pose = left_arm.get_current_pose().pose

    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = current_pose.position.x
    pose_goal.position.y = current_pose.position.y
    pose_goal.position.z = current_pose.position.z
    pose_goal.orientation.x = -0.48673041075
    pose_goal.orientation.y = 0.509701024424
    pose_goal.orientation.z = 0.501773446261
    pose_goal.orientation.w = 0.501519472782

    left_arm.set_pose_target(pose_goal)
    left_arm.go(wait=True)
    left_arm.stop()
    left_arm.clear_pose_targets()

    time.sleep(0.5)


def reset_arm():
    move_arm(*ROBOT_ORIGIN)
    gripper_down()


def move_arm(x, y, z=None):
    old_pose = left_arm.get_current_pose().pose

    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = x
    pose_goal.position.y = y
    if z is None:
        pose_goal.position.z = old_pose.position.z
    else:
        pose_goal.position.z = z
    pose_goal.orientation.x = old_pose.orientation.x
    pose_goal.orientation.y = old_pose.orientation.y
    pose_goal.orientation.z = old_pose.orientation.z
    pose_goal.orientation.w = old_pose.orientation.w

    left_arm.set_pose_target(pose_goal)
    success = left_arm.go(wait=True)
    left_arm.stop()
    left_arm.clear_pose_targets()

    current_pose = left_arm.get_current_pose().pose
    x_err = np.abs(pose_goal.position.x - current_pose.position.x)
    y_err = np.abs(pose_goal.position.y - current_pose.position.y)
    z_err = np.abs(pose_goal.position.z - current_pose.position.z)
    total_err = np.sqrt(x_err ** 2 + y_err ** 2 + z_err ** 2)

    print('py2::Error: {}'.format(total_err))
    if not success:
        print('py2::Action failed...')


def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)


def image_callback(msg):
    global send_message
    global waiting_to_send_image

    if send_message:
        send_message = False
        bridge = cv_bridge.CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        exposure_img = gamma_trans(image, .6)
        # path = os.path.join('images', 'exposure_test.png')
        # cv2.imwrite(path, exposure_img)
        
        data = pickle.dumps(exposure_img)
        data = zlib.compress(data)
        socket.send(data)
        print('py2::Image sent')
        waiting_to_send_image = False


def execute_action(location, delta):
    start_loc, end_loc = location, location + delta * MAX_IMAGE_DELTA
    start_loc = coord_image_to_robot(start_loc)
    end_loc = coord_image_to_robot(end_loc)

    print('pr2::Moving from pose {} to {}...'.format(start_loc, end_loc))

    move_arm(start_loc[0], start_loc[1], Z_STATIONARY)
    time.sleep(0.5)
    move_arm(start_loc[0], start_loc[1], Z_DOWN)
    time.sleep(0.5)
    close_gripper()
    time.sleep(2.5) # wait longer since close doesn't block
    move_arm(start_loc[0], start_loc[1], Z_UP)
    time.sleep(0.5)
    move_arm(end_loc[0], end_loc[1], Z_UP)
    time.sleep(0.5)
    move_arm(end_loc[0], end_loc[1], Z_DOWN)
    time.sleep(0.5)
    open_gripper()
    time.sleep(2.5) # wait longer since open doesn't block
    move_arm(end_loc[0], end_loc[1], Z_UP)
    time.sleep(0.5)

    reset_arm()
    time.sleep(2.5)
    print('pr2::Completed action')


def coord_image_to_robot(image_coord):
    image_coord += 0.5
    image_coord *= float(IMAGE_SIZE) / IMAGE_INPUT_SIZE
    image_coord += IMAGE_ORIGIN.astype('float32')
    return image_coord.dot(A) + b


if __name__ == '__main__':
    socket = init_socket()
    pub, left_arm = init_controller()

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
        action = pickle.loads(data)

        print('pr2::Timestep {}'.format(time_step))
        print('pr2::Received action {}'.format(action))
        location, delta = action[:2], action[2:]
        assert len(location) == len(delta) == 2

        execute_action(location, delta)
        send_message = True
        waiting_to_send_image = True
        time_step += 1
