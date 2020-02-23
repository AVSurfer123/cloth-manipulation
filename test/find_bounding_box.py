import rospy
from sensor_msgs.msg import Image
import cv_bridge
import cv2
from os.path import join

called = False
image_name = None

s = 300
r, c = 125, 250

def callback(msg):
    global called

    if not called:
        called = True
        bridge = cv_bridge.CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image[r:r+s, c:c+s, :]
        # image = cv2.resize(image, (64, 64))

        if image_name is not None:
            path = join('images', image_name)
            cv2.imwrite(path, image)
            print('Saved:', path)


def main():
    global called
    global image_name
    global r
    global c
    global s
    rospy.init_node('test_rostopic')
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback)

    try:
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
