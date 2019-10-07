import rospy
from sensor_msgs.msg import Image
import cv_bridge

global called
called = False
def callback(msg):
    global called
    if not called:
        called = True
        print(msg.height, msg.width, len(msg.data))
        bridge = cv_bridge.CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        print(image)

def main():
    rospy.init_node('test_rostopic')
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    main()