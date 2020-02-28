import rospy
from sensor_msgs.msg import Image
import cv_bridge
import cv2
import signal
import sys
import numpy as np


def callback(msg):
    bridge = cv_bridge.CvBridge()
    image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    cv2.imshow('Image', image)
    cv2.waitKey(1)

def signal_handler(signal, frame):
    print('\npy2::Ending process')
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node('display_image')
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
