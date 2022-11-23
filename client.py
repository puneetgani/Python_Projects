# import socket
#
# c = socket.socket()
#
# c.connect(('localhost',10766))
# name = input("Enter your name: ")
# c.send(bytes(name,'utf-8'))
#
# print(c.recv(1024).decode())
# c.close()

import socket
import json

ip = 'localhost'
port = 55556

c = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print("client socket created")
c.connect(( ip , port))

username = input("enter username: ")
password = input("enter password: ")
username_password = {'username': username , 'password': password}
username_passwords = json.dumps(username_password)

c.send(bytes(username_passwords,'utf-8'))
# password = input("enter password: ")
# c.send(bytes(password,'utf-8'))
print(c.recv(1024).decode())
c.close()