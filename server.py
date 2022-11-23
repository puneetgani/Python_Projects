# import socket
#
# s = socket.socket()
# print("socket created")
#
# s.bind(("localhost",10766)) # to bind the particular socket to a IP and a Port number : it provides service from that port
#
# s.listen(3) # is used to listen to 3 clients
# print("waiting for connection")
#
# while True: # infinte loop
#     c, addr = (s.accept())
#     name = c.recv(1024).decode()
#     print("connected with",addr,name)
#
#     c.send(bytes("welcome to puneet's chat",'utf-8'))
#     c.close()


import socket
import json

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print("server socket created")
s.bind(('localhost',55556))
s.listen(3)
print("waiting for connection...")

while True:
    c, addr = s.accept()
    print("connected with: ",addr)
    username_password = c.recv(1024).decode()
    username_passwords = json.loads(username_password)

    if username_passwords.get('username') == 'heartcoder' and username_passwords.get('password') == 'lovecoding':
       c.send(bytes("okay",'utf-8'))
    else:
       c.send(bytes("fail",'utf-8'))
