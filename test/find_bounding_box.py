import rospy
from sensor_msgs.msg import Image
import cv_bridge
import cv2
import os
import sys
import signal
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from constants import TEST_GRIPPER_POS as GRIPPER_POS, IMAGE_ORIGIN, IMAGE_SIZE
from vision_utils import gamma_trans


called = False
image_name = None

s = IMAGE_SIZE
r, c = IMAGE_ORIGIN

def callback(msg):
    global called

    if not called:
        # called = True
        bridge = cv_bridge.CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        image = image[r:r+s, c:c+s, :]
        image = gamma_trans(image, 1.)
        cv2.imshow('Image', image)
        cv2.waitKey(1)
        # image = cv2.resize(image, (64, 64))

        # if image_name is not None:
        #     path = os.path.join('images', image_name)
        #     cv2.imwrite(path, image)
        #     print('Saved:', path)

def signal_handler(signal, frame):
    print('\npy2::Ending process')
    sys.exit(0)


def main():
    global called
    global image_name
    global r
    global c
    global s
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node('find_bounding_box')
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback)

    try:
        # rospy.spin()
        while True:
            print('Press enter row column size:')
            string = raw_input()
            if string != "":
                r, c, s = raw_input().split()
                r, c, s = int(r), int(c), int(s)
            image_name = 'tmp.png'
            called = False
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
