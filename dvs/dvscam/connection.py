class Serial(object):
    def __init__(self, port, baud):
        import serial
        self.conn = serial.Serial(port, baudrate=baud, rtscts=True, timeout=0)
    def send(self, message):
        self.conn.write(message)
    def receive(self):
        return self.conn.read(1024)
    def close(self):
        self.conn.close()

import socket
class Socket(object):
    cache = {}
    def __init__(self, address, port=56000):
        self.socket = Socket.get_socket(address, port)

    @classmethod
    def get_socket(cls, address, port):
        key = (address, port)
        s = cls.cache.get(key, None)
        if s is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((address, port))
            s.settimeout(0)
            cls.cache[key] = s
        return s

    def send(self, message):
        self.socket.send(message)
    def receive(self):
        try:
            return self.socket.recv(1024)
        except socket.error:
            return ''
    def close(self):
        self.socket.close()

import cPickle as pickle
import time
class Recording(object):
    def __init__(self, fname):
        self.data = pickle.load(open(fname, 'rb'))
        self.start_time = time.time()
        self.time_elapsed = 0
        self.index = 0 # current index into the data list
    def send(self, message):
        pass
    def receive(self):
        if time.time() - self.start_time >= self.data[self.index][0]:
            self.index += 1
            return self.data[self.index-1][1]
        else:
            return ''
    def close(self):
        pass

