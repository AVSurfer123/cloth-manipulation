import zmq
import time
import sys

context = zmq.Context()
print "Connecting to server..."
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:%s" % 7778)

for request in range (1,10):
    print "Sending request ", request,"..."
    socket.send ("Hello")
    #  Get the reply.
    message = socket.recv()
    print "Received reply ", request, "[", message, "]"

