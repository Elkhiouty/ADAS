import socket

def send(t):
    print(t)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp = ('172.20.10.13', 2390)
    
    if t == 'Turn Left' :
        s.sendto('l'.encode(),udp)
    elif t == 'Turn Right' :
        s.sendto('r'.encode(),udp)
    elif t == 'Bump' :
        s.sendto('b'.encode(),udp)
    elif t == 'Stop' :
        s.sendto('D'.encode(),udp)
    elif t == 'Speed Limit 20' :
        s.sendto('s20'.encode(),udp)
    elif t == 'Speed Limit 30' :
        s.sendto('s30'.encode(),udp)
    elif t == 'Speed Limit 50' :
        s.sendto('s50'.encode(),udp)
    elif t == 'Speed Limit 60' :
        s.sendto('s60'.encode(),udp)
    elif t == 'Speed Limit 70' :
        s.sendto('s70'.encode(),udp)
    elif t == 'Speed Limit 80' :
        s.sendto('s80'.encode(),udp)
    elif t == 'other':
        t == 'other'
    else :
        s.sendto(t.encode(),udp)
    

