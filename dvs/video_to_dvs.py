# Convert Video from ROS bagfiles into an approximate DVS recording

import rospy
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import deque

import cv2

from sensor_msgs.msg import Image, CompressedImage

class ImageViewer(object):
    def __init__(self, root='/irat_red'):
        self.root = root

        self.im_data = deque() # Captured Images

        # Set up regular image Figure
        self.im_fig = plt.figure(1)
        self.im_ax = self.im_fig.add_subplot(111)
        self.im_ax.set_title("Video")
        #self.im_im = self.im_ax.imshow(np.zeros((256, 256, 4))) # Blank starting image
        self.im_im = self.im_ax.imshow(np.zeros((240, 416, 3))) # Blank starting image
        self.im_fig.show()
        self.im_im.axes.figure.canvas.draw()
        
        # Set up DVS Figure
        self.dv_fig = plt.figure(2)
        self.dv_ax = self.im_fig.add_subplot(111)
        self.dv_ax.set_title("DVS Video")
        self.dv_im = self.im_ax.imshow(np.zeros((256, 256)),cmap=plt.cm.gray) # Blank starting image
        self.dv_fig.show()
        self.dv_im.axes.figure.canvas.draw()

        rospy.init_node('image_viewer', anonymous=True)

        self.sub_im = rospy.Subscriber( self.root + '/camera/image/compressed',
                                        CompressedImage,
                                        self.im_callback
                                      )

    def run(self):
        while not rospy.is_shutdown():
            if self.im_data:
                im = self.im_data.popleft()
                self.im_im.set_data(im)
                self.im_im.axes.figure.canvas.draw()

    def im_callback(self, data):
        np_arr = np.fromstring(data.data, np.uint8)
        im = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        self.im_data.append( im )

if __name__ == '__main__':
    viewer = ImageViewer()
    viewer.run()
