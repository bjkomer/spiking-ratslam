import numpy as np
import dvscam # modified copy of nstbot

import time

bot = dvscam.RetinaBot()
bot.connect(dvscam.Recording('recording6.p'))

time.sleep(1)
bot.retina(True)

bot.show_image()

while True:
    time.sleep(1)
