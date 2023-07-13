import serial
from time import sleep

def openc():
    global ser
    ser = serial.Serial (port="/dev/ttyS0",baudrate=9600,stopbits=serial.STOPBITS_ONE,
                         bytesize=serial.EIGHTBITS)    #Open port with baud rate
    sleep(2)

def send(t):
    if not(ser.isOpen()) :
        openc()
    if t == 'Turn Left' :
        ser.write('l'.encode())
    elif t == 'Turn Right' :
        ser.write('r'.encode())
    elif t == 'Bump' :
        ser.write('b'.encode())
    elif t == 'Stop' :
        ser.write('s0'.encode())
    elif t == 'Speed Limit 20' :
        ser.write('s20'.encode())
    elif t == 'Speed Limit 30' :
        ser.write('s30'.encode())
    elif t == 'Speed Limit 50' :
        ser.write('s50'.encode())
    elif t == 'Speed Limit 60' :
        ser.write('s60'.encode())
    elif t == 'Speed Limit 70' :
        ser.write('s70'.encode())
    elif t == 'Speed Limit 80' :
        ser.write('s80'.encode())
    elif t == 'other':
        t == 'other'
    else :
        ser.write(t.encode())
    

