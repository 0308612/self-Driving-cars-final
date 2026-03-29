import serial
import time
import random

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    ser.reset_input_buffer()

    while True:
        a = random.randint(1,2)
        if a == 1:
            ser.write(b"motor go\n")
        elif a == 2:
            ser.write(b"motor off\n")
        #decode and read and print to RP console
        line = ser.readline().decode('utf-8').rstrip()
        print(line)
        time.sleep(1)