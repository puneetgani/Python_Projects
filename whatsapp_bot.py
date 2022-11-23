import pywhatkit
import time

l = ["+919980170863"]
k = 52
for i in l:
    pywhatkit.sendwhatmsg(i ,"hello ",23,k)
    time.sleep(3)
    k += 1
