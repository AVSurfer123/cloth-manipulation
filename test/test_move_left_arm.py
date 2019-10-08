import sys
import os
import rospy
import numpy as np
import moveit_commander
import geometry_msgs.msg
from pr2_controllers_msgs.msg import Pr2GripperCommand
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from constants import TEST_GRIPPER_POS as GRIPPER_POS

TOPIC_NAME = '/l_gripper_controller/command'
pub_gripper = rospy.Publisher(TOPIC_NAME, Pr2GripperCommand, queue_size=10)


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('pr2_move_left_arm_test')
robot = moveit_commander.RobotCommander()

left_arm = moveit_commander.MoveGroupCommander('left_arm')
left_gripper = moveit_commander.MoveGroupCommander('left_gripper')

group_names = robot.get_group_names()

print('Available Planning Groups:', robot.get_group_names())
print('Current state:', robot.get_current_state())

def left_pose_pos(x, y, z):
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = x
    pose_goal.position.y = y
    pose_goal.position.z = z
    pose_goal.orientation.x = GRIPPER_POS[0]
    pose_goal.orientation.y = GRIPPER_POS[1]
    pose_goal.orientation.z = GRIPPER_POS[2]
    pose_goal.orientation.w = GRIPPER_POS[3]

    left_arm.set_pose_target(pose_goal)
    left_arm.go(wait=True)
    left_arm.stop()
    left_arm.clear_pose_targets()

    current_pose = left_arm.get_current_pose().pose
    x_err = np.abs(pose_goal.position.x - current_pose.position.x)
    y_err = np.abs(pose_goal.position.y - current_pose.position.y)
    z_err = np.abs(pose_goal.position.z - current_pose.position.z)
    print('Error:', np.sqrt(x_err ** 2 + y_err ** 2 + z_err ** 2))
    print(x_err, y_err, z_err)


def left_pose_ori(x, y, z, w):
    current_pose = left_arm.get_current_pose().pose

    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = current_pose.position.x
    pose_goal.position.y = current_pose.position.y
    pose_goal.position.z = current_pose.position.z
    pose_goal.orientation.x = x
    pose_goal.orientation.y = y
    pose_goal.orientation.z = z
    pose_goal.orientation.w = w

    left_arm.set_pose_target(pose_goal)
    plan = left_arm.plan()
    print plan
    left_arm.execute(plan, wait=True)
    # left_arm.go(wait=True)
    left_arm.stop()
    left_arm.clear_pose_targets()


def left_move_joint(joint_values):
    left_arm.set_joint_value_target(joint_values)
    left_arm.go(wait=True)
    left_arm.stop()



def left_open():
    pub_gripper.publish(Pr2GripperCommand(0.025, 32))


def left_close():
    pub_gripper.publish(Pr2GripperCommand(0.0, 32))

def left_gripper_down():
    left_pose_ori(*GRIPPER_POS)
