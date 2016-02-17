import retinabot
import numpy as np

class RetinaDebugBot(retinabot.RetinaBot):
    def initialize(self):
        super(RetinaDebugBot, self).initialize()
        self.x = None
        self.y = None
        self.t = None
        self.p = None
        self.last_off = None
        self.delta = None

    def process_retina(self, data):
        super(RetinaDebugBot, self).process_retina(data)
        packet_size = self.retina_packet_size
        y = data[::packet_size] & 0x7f
        x = data[1::packet_size] & 0x7f
        t = data[2::packet_size].astype(np.uint32)
        t = (t << 8) + data[3::packet_size]
        if packet_size >= 5:
            t = (t << 8) + data[4::packet_size]
        if packet_size >=6:
            t = (t << 8) + data[5::packet_size]
        on = (data[1::packet_size] & 0x80) > 0

        if self.x is None:
            self.x = x
            self.y = y
            self.t = t
            self.on = on
            self.last_off = np.zeros((128, 128), dtype=np.uint32)
        else:
            indexes = np.logical_and(on==0, self.last_off[x, y] != 0)
            delta = t[indexes] - self.last_off[x[indexes], y[indexes]]
            self.last_off[x[on==0], y[on==0]] = t[on==0]
            if self.delta is None:
                self.delta = delta
            else:
                self.delta = np.hstack([self.delta, delta])

            self.x = np.hstack([self.x, x])
            self.y = np.hstack([self.y, y])
            self.t = np.hstack([self.t, t])
            self.on = np.hstack([self.on, on])


    def data_loop(self):
        import pylab
        fig = pylab.figure()
        pylab.ion()

        while True:
            fig.clear()
            #pylab.plot(self.t[np.where(self.on==0)])
            hz = 1000000 / self.delta 
            pylab.hist(hz, 50, range=(800, 1200))
            pylab.xlim(500, 1500)
            pylab.ylim(0, 100)
            self.delta = self.delta[:0]

            fig.canvas.draw()
            fig.canvas.flush_events()


        


if __name__ == '__main__':
    import connection
    bot = RetinaDebugBot()
    bot.connect(connection.Serial('/dev/ttyUSB0', baud=4000000))
    bot.retina(True)
    #bot.show_image()

    import thread
    thread.start_new_thread(bot.data_loop, ())

    import time
    while True:
        time.sleep(1)






