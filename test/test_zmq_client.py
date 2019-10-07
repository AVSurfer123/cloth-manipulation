import zmq
import time
import sys

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % 7778)

while True:
    #  Wait for next request from client
    message = socket.recv()
    print "Received request: ", message
    time.sleep (1)
    socket.send("World from %s" % 7778)