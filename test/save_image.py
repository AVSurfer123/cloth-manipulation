import rospy
from sensor_msgs.msg import Image
import cv_bridge
import cv2
from os.path import join

called = False
image_name = None

def callback(msg):
    global called

    if not called:
        called = True
        bridge = cv_bridge.CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        s = 300
        r, c = 130, 175
        image = image[r:r+s, c:c+s, :]
        image = cv2.resize(image, (64, 64))

        if image_name is not None:
            path = join('images', image_name)
            cv2.imwrite(path, image)
            print 'Saved:', path


def main():
    global called
    global image_name
    rospy.init_node('test_rostopic')
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
