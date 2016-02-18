import time
import atexit

from . import connection

class NSTBot(object):
    def __init__(self):
        self.connection = None
        self.recording = False
        self.events = [] # a list recording all events to be saved

    def connect(self, connection):
        self.connection = connection
        self.last_time = {}
        self.initialize()
        atexit.register(self.disconnect)

    def send(self, key, message, msg_period=None):
        now = time.time()
        if msg_period is None or now > self.last_time.get(key, 0) + msg_period:
            self.connection.send(message)
            self.last_time[key] = now

    def receive(self):
        return self.connection.receive()

    def initialize(self):
        self.connection.send('\n')

    def disconnect(self):
        self.connection.close()



