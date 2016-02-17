import numpy as np
import dvscam # modified copy of nstbot

import time

#bot = dvscam.RetinaBot()
bot = dvscam.MultiCam()
bot.connect(dvscam.Serial('/dev/ttyUSB0', baud=4000000))

time.sleep(1)
bot.retina(True)

decay_field = np.ones((128,128))
"""
decay_field[10:-10,10:-10] = .9
decay_field[20:-20,20:-20] = .7
decay_field[30:-30,30:-30] = .5
decay_field[40:-40,40:-40] = .3
decay_field[50:-50,50:-50] = .1
"""

m = np.sqrt(64**2 + 64**2)

def decay_func(i,j):
    r = np.sqrt((i-64)**2 + (j-64)**2)
    return r/(m+1)

for i in range(128):
    for j in range(128):
        decay_field[i,j] = decay_func(i,j)
print(decay_field)

#bot.show_image(decay=decay_field) #default decay=0.5
#bot.show_images(decays=[0.1, 0.5])
bot.show_images(decays=[0.1, 0.5, 0.9, decay_field])

while True:
    time.sleep(1)
