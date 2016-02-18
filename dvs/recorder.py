import numpy as np
import dvscam # modified copy of nstbot

import time

bot = dvscam.RetinaBot()
bot.record('recording6.p')
bot.connect(dvscam.Serial('/dev/ttyUSB0', baud=4000000))

time.sleep(1)
bot.retina(True)

bot.show_image()

while True:
    time.sleep(1)
