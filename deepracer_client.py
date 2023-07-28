import socket
import struct
import atexit
import time
import numpy as np
import cv2

import time
import functools
import logging
import os
import threading

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

from ctypes import *
PWM_CONTROL_PROGRAM = "/home/deepracer/pwm_ctrl.so"
pwm_ctrl = CDLL(PWM_CONTROL_PROGRAM)

SERVER_IP = '128.194.50.149'
PORT = 65432
INT_BYTE_LIMIT = 4

# Calibration values
# These values were good for a few hours...
# SERVO_MAX = 1700000
# SERVO_MIN = 1200000
# SERVO_MID = 1400000
# MOTOR_MAX = 1500000
# MOTOR_MIN = 1250000
# MOTOR_MID = 1350000
SERVO_MAX = 1700000
SERVO_MID = 1500000
SERVO_MIN = 1200000
MOTOR_MAX = 1550000
MOTOR_MID = 1450000
MOTOR_MIN = 1340000

POLARITY_SERVO_VAL = 1
POLARITY_MOTOR_VAL = -1
SERVO_PERIOD = 20000000


def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        diff = round(time.time() - start_time, 3)
        logging.debug(f"{func.__name__} time: {str(diff)}")
        return result
    return wrapper


@log
def get_image(camera):
    if camera is None:
        return np.array([[0, 1], [2, 3]])
    rval, image = camera.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale_percent = 20 # percent of original size
    height = int(image.shape[0] * scale_percent / 100)
    width = int(image.shape[1] * scale_percent / 100)
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    image = cv2.imencode('.jpg', image)[1]
    # TODO pad image so its the same size every time
    #cv2.imwrite('frame.jpg', image)
    return image

@log
def send_image(conn, image):
    image = np.array(image).tobytes()
    size = len(image)
    size = size.to_bytes(INT_BYTE_LIMIT, 'big')
    conn.sendall(size)
    conn.sendall(image)


@log
def send_battery(conn, batt_lvl):
    packed = struct.pack("!f", batt_lvl)
    conn.sendall(packed)


@log
def recv_action(conn):
    steering = struct.unpack("!f", conn.recv(INT_BYTE_LIMIT))[0]
    throttle = struct.unpack("!f", conn.recv(INT_BYTE_LIMIT))[0]

    return steering, throttle


@log
def set_max_speed(battery):
    if battery == 11:
        MAX_FORWARD = 0.60
        MAX_REVERSE = 0.60
    elif battery == 10:
        MAX_FORWARD = 0.70
        MAX_REVERSE = 0.70
    elif battery == 9:
        MAX_FORWARD = 0.72
        MAX_REVERSE = 0.72
    elif battery == 5:
        MAX_FORWARD = 0.82
        MAX_REVERSE = 0.82
    elif battery == 4:
        MAX_FORWARD = 0.85
        MAX_REVERSE = 0.85
    else:
        MAX_FORWARD = 0.80
        MAX_REVERSE = 0.80
    return MAX_FORWARD, MAX_REVERSE

@log
def compute_pwm_value(action, polarity, minv, midv, maxv):
    action = np.clip(action, -1.0, 1.0)
    adjVal = action * polarity
    ret_val = None
    if adjVal < 0:
        ret_val = midv + adjVal * (midv - minv)
    elif adjVal > 0:
        ret_val = midv + adjVal * (maxv - minv)
    else:
        ret_val = midv
    return int(ret_val)


@log
def set_motor(action):
    pwmv = compute_pwm_value(action, POLARITY_MOTOR_VAL, MOTOR_MIN, MOTOR_MID, MOTOR_MAX)
    #duty = "/sys/class/pwm/pwmchip0/pwm0/duty_cycle"
    #os.system('echo ' + str(pwmv) + ' > ' + duty)
    pwm_ctrl.writePWM(0, pwmv)


@log
def set_steering(action):
    pwmv = compute_pwm_value(action, POLARITY_SERVO_VAL, SERVO_MIN, SERVO_MID, SERVO_MAX)
    #duty = "/sys/class/pwm/pwmchip0/pwm1/duty_cycle"
    #os.system('echo ' + str(pwmv) + ' > ' + duty)
    pwm_ctrl.writePWM(1, pwmv)


@log
def timed_sleep():
    time.sleep(0.05)


@log
def vehicle_operation(camera, conn):
    image = get_image(camera)
    send_image(conn, image)
    # send_battery(conn, batt_lvl)
    steering, throttle = recv_action(conn)
    logging.debug('Recvd steer: ' + str(round(steering, 3)) + ' throt: ' + str(round(throttle, 3)))
    
    #timed_sleep()
    set_steering(steering)
    set_motor(throttle)
    #timed_sleep()
    


def stop_vehicle():
    pwm_ctrl.writePWM(0, 0)


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
    

class AsyncCamera():
    def __init__(self):
        self.camera = cv2.VideoCapture()
        self.camera.open(0)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.camera_thread = StoppableThread(target=self.update, args=())
        self.camera_thread.daemon = True
        self.camera_thread.start()
        # camera.open(0, apiPreference=cv2.CAP_V4L2)
        # camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
        # camera.set(cv2.CAP_PROP_FPS, 30.0)
        self.status = 0
        self.frame = None
        time.sleep(5)

    def update(self):
        while True:
            if self.camera.isOpened():
                (self.status, self.frame) = self.camera.read()
            time.sleep(1/30)    # Frame rate is 30

    def read(self):
        return self.status, self.frame
    
    def release(self):
        self.camera_thread.stop()
        self.camera_thread.join()
        self.camera.release()


@log
def main(args=None):
    atexit.register(stop_vehicle)
    while (True):
        conn = None
        camera = None
        try:
            print('Waiting for connection to server')
            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            while True:
                try:
                    conn.connect((SERVER_IP, PORT))
                    break
                except socket.error as err:
                    logging.debug('Connection attempted: ' + str(err))
                    time.sleep(1)

            print('Starting camera')
            camera = AsyncCamera()

            print('Ready')
            while True:
                vehicle_operation(camera, conn)
        except Exception as e:
            print(e)
            
            if conn is not None: conn.close()
            if camera is not None: camera.release()
            conn, camera = None, None


if __name__ == '__main__':
    main()
