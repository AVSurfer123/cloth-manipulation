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
from robot_actions import *


def execute_action(picks, deltas, two_hand):
    if two_hand:
        left_loc, left_delta, right_loc, right_delta = picks[0], deltas[0], picks[1], deltas[1]

        left_start, left_end = left_loc, left_loc + left_delta * MAX_IMAGE_DELTA
        right_start, right_end = right_loc, right_loc + right_delta * MAX_IMAGE_DELTA
        left_start, left_end, right_start, right_end = map(coord_image_to_robot, [left_start, left_end, right_start, right_end])

        print('pr2::Moving left from pose {} to {}...'.format(left_start, left_end))
        print('pr2::Moving right from pose {} to {}...'.format(right_start, right_end))

        success = move_both_arms((left_start[0], left_start[1], Z_STATIONARY), (right_start[0], right_start[1], Z_STATIONARY))
        if not success:
            print('pr2::Failed to find a execution plan, skipping action.')
            return
        time.sleep(0.5)
        move_both_arms((left_start[0], left_start[1], Z_DOWN), (right_start[0], right_start[1], Z_DOWN + RIGHT_Z_OFFSET))
        time.sleep(0.5)
        close_gripper('left')
        close_gripper('right')
        time.sleep(2.5)  # wait longer since close doesn't block
        move_both_arms((left_start[0], left_start[1], Z_UP), (right_start[0], right_start[1], Z_UP))
        time.sleep(0.5)
        move_both_arms((left_end[0], left_end[1], Z_UP), (right_end[0], right_end[1], Z_UP))
        time.sleep(0.5)
        move_both_arms((left_end[0], left_end[1], Z_DOWN), (right_end[0], right_end[1], Z_DOWN + RIGHT_Z_OFFSET))
        time.sleep(0.5)
        open_gripper('right')
        open_gripper('left')
        time.sleep(2.5)  # wait longer since open doesn't block
        move_both_arms((left_end[0], left_end[1], Z_UP), (right_end[0], right_end[1], Z_UP))
        time.sleep(0.5)

        reset_both_arms()
        time.sleep(1)

    else:
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
        
        data = pickle.dumps(exposure_img)
        data = zlib.compress(data)
        socket.send(data)
        print('py2::Image sent')
        waiting_to_send_image = False
        print("In callback:", waiting_to_send_image)


if __name__ == "__main__":
    socket = init_socket()
    main_side = 'right'  # Choose the side to use if one-handed

    rospy.init_node('towel_folding_py2')
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


