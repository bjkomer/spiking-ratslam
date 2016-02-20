# Convert Video from ROS bagfiles into an approximate DVS recording

import rospy
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import deque

import cv2

from sensor_msgs.msg import Image, CompressedImage

# Height and width of images
H = 240
W = 416

class ImageViewer(object):
    def __init__(self, root='/irat_red'):
        self.root = root

        self.im_data = deque() # Captured Images
        self.prev_im = np.zeros((H, W), dtype=np.int16)+128
        self.tol = 10 # tolerance for detecting a change

        # Set up regular image Figure
        self.im_fig = plt.figure(1)
        self.im_ax = self.im_fig.add_subplot(111)
        self.im_ax.set_title("Video")
        #self.im_im = self.im_ax.imshow(np.zeros((256, 256, 4))) # Blank starting image
        self.im_im = self.im_ax.imshow(np.zeros((H, W, 3))) # Blank starting image
        self.im_fig.show()
        self.im_fig.canvas.draw()
        
        # Set up DVS Figure
        
        self.dv_fig = plt.figure(2)
        self.dv_ax = self.dv_fig.add_subplot(111)
        self.dv_ax.set_title("DVS Video")
        self.dv_im = self.dv_ax.imshow(np.zeros((H, W)),
                                       #vmax=255, vmin=-255,
                                       vmax=1, vmin=-1,
                                       cmap=plt.cm.gray) # Blank starting image
        self.dv_fig.show()
        self.dv_fig.canvas.draw()
        
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
                self.im_fig.canvas.draw()
                
                # Get grayscale image for converting to DVS
                gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                diff_im = gray_im.astype(np.int16) - self.prev_im
                diff_im[np.where((diff_im < self.tol) & (diff_im > -self.tol))] = 0
                diff_im[np.where(diff_im >= self.tol)] = 1
                diff_im[np.where(diff_im <= -self.tol)] = -1
                self.prev_im = np.copy(gray_im)
                self.dv_im.set_data(diff_im)
                self.dv_fig.canvas.draw()

    def im_callback(self, data):
        np_arr = np.fromstring(data.data, np.uint8)
        #im = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        bgr_im = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)

        # Need to convert BGR image to RGB
        b,g,r = cv2.split(bgr_im)
        rgb_im = cv2.merge([r,g,b])
        
        self.im_data.append( rgb_im )

if __name__ == '__main__':
    viewer = ImageViewer()
    viewer.run()
