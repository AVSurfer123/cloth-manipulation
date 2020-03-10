import rospy
from sensor_msgs.msg import Image
import cv_bridge
import cv2
import os
import signal
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from constants import TEST_GRIPPER_POS as GRIPPER_POS, IMAGE_ORIGIN, IMAGE_SIZE, GAMMA_CORRECTION
from vision_utils import gamma_trans

called = False
image_name = None

def callback(msg):
    global called

    if not called:
        called = True
        bridge = cv_bridge.CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        s = IMAGE_SIZE
        r, c = IMAGE_ORIGIN
        image = image[r:r+s, c:c+s, :]
        image = cv2.resize(image, (64, 64))
        image = gamma_trans(image, GAMMA_CORRECTION)

        if image_name is not None:
            path = os.path.join('images', image_name)
            cv2.imwrite(path, image)
            print('Saved:', path)

def signal_handler(signal, frame):
    print('\npy2::Ending process')
    sys.exit(0)

def main():
    global called
    global image_name
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node('save_imaage')
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback)

    try:
        while True:
            print('Press Enter')
            name = raw_input()
            image_name = '{}.png'.format(name)
            called = False
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
