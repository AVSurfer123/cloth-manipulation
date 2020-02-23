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
from robot_actions import *


def gripper_down_backup():
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


def execute_action(picks, deltas, two_hand):
    if not two_hand:
        location, delta = picks[0], deltas[0]
        start_loc, end_loc = location, location + delta * MAX_IMAGE_DELTA
        start_loc = coord_image_to_robot(start_loc)
        end_loc = coord_image_to_robot(end_loc)

        print('pr2::Moving from pose {} to {}...'.format(start_loc, end_loc))

        move_arm(main_side, (start_loc[0], start_loc[1], Z_STATIONARY))
        time.sleep(0.5)
        move_arm(main_side, (start_loc[0], start_loc[1], Z_DOWN))
        time.sleep(0.5)
        close_gripper(main_side)
        time.sleep(2.5) # wait longer since close doesn't block
        move_arm(main_side, (start_loc[0], start_loc[1], Z_UP))
        time.sleep(0.5)
        move_arm(main_side, (end_loc[0], end_loc[1], Z_UP))
        time.sleep(0.5)
        move_arm(main_side, (end_loc[0], end_loc[1], Z_DOWN))
        time.sleep(0.5)
        open_gripper(main_side)
        time.sleep(2.5) # wait longer since open doesn't block
        move_arm(main_side, (end_loc[0], end_loc[1], Z_UP))
        time.sleep(0.5)

        reset_arm(main_side)
        time.sleep(1)
    print('pr2::Completed action')
    
send_message = True
waiting_to_send_image = True

def image_callback(socket, msg):
    global send_message, waiting_to_send_image

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


if __name__ == '__main__':
    socket = init_socket()
    main_side = 'right'  # Choose the side to use if one-handed

    rospy.init_node('rope_folding_py2')
    rospy.Subscriber("/camera/rgb/image_raw", Image, lambda msg: image_callback(socket, msg))

    print('py2::Starting...')
    time_step = 0
    open_gripper('left')
    open_gripper('right')
    reset_both_arms()
    time.sleep(1)
    while time_step < N_ACTIONS:
        if waiting_to_send_image:
            continue

        data = socket.recv()
        data = zlib.decompress(data)
        picks, deltas, two_hand = pickle.loads(data)

        print('pr2::Timestep {}'.format(time_step))
        print('pr2::Received action {}'.format((picks, deltas)))
        if two_hand:
            assert len(picks) == len(deltas) == 2
        else:
            assert len(picks) == len(deltas) == 1
            location, delta = picks[0], deltas[0]
            assert len(location) == len(delta) == 2

        execute_action(picks.astype('double'), deltas.astype('double'), two_hand)
        send_message = True
        waiting_to_send_image = True
        time_step += 1
