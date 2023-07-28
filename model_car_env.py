from config import *

import time
import functools
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import socket
import struct
import atexit
import numpy as np
import math
import cv2

from shapely.geometry import Point, Polygon
from shapely import affinity
from centerline.geometry import Centerline
import asyncio
from websockets.sync.client import connect
#import blissful_basics as bb
import time
import os
import subprocess
from pynput import keyboard
from pynput.keyboard import Key
import threading
from concurrent.futures.thread import ThreadPoolExecutor

import gymnasium as gym
from gymnasium import spaces
from torchvision import transforms
import torch


this_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])

''' LOAD TRACK AND BORDER '''
# Load maps
def load_map(filename, offset_x, offset_y):
    track = []
    track_coords = open(filename, 'r')

    for coord in track_coords:
        t = coord.split(',')
        track.append((float(t[0])+offset_x,float(t[1])+offset_y))
    
    track_coords.close()
    return track

# Fix rotation of track map
def fix_map_rotation(track,bounds_map,offset_x,offset_y,error_threshold=1):
    bounds_map_calib=[(0.024254 , -1.776745),(0.58907 , 0.523119),(4.273964 , -0.46625),(3.576584 , -2.787911)] #DONT CHANGE
    coords = [(np.round(coord[0]+offset_x,5),np.round(coord[1]+offset_y,5)) for coord in bounds_map_calib]
    bounds_map_calib = Polygon(coords)
    low_percent_error = 100
    rotation_degree = 0

    # Find best rotation angle
    for x in range(361):
        coords_calib = list(affinity.rotate(bounds_map_calib, x, bounds_map[0]).exterior.coords)

        x_percent_error = np.abs(((coords_calib[2][0]-bounds_map[2][0])/bounds_map[2][0]))
        y_percent_error = np.abs(((coords_calib[2][1]-bounds_map[2][1])/bounds_map[2][1]))

        if (x_percent_error+y_percent_error)/2 < low_percent_error:
            low_percent_error = (x_percent_error+y_percent_error)/2
            #print('coords calib:',coords_calib,'| bounds_map:',(bounds_map[2][0],bounds_map[2][1]))
            #print(x_percent_error,y_percent_error)
            rotation_degree = x
    
    if low_percent_error > error_threshold:
        print('Warning: Track Rotation is: ',np.round(low_percent_error*100,5),'%',sep='')
    else: print('Track Percent Error: ',np.round(low_percent_error*100,5),'%',sep='')
    return affinity.rotate(track,rotation_degree,bounds_map[0])

coords = [  (-0.744893, 5.563468),
            (0.070752, 3.315458), 
            (-3.45909, 2.046855), 
            (-4.247611, 4.292663)]
center_coord = (-2.088408, 3.821089)

offset_x = coords[0][0] - (0.024254)
offset_y = coords[0][1] - (-1.776745)

track_outer = load_map('/home/pistar/Desktop/JetRacer/trackouter.log',offset_x,offset_y)
track_inner1 = load_map('/home/pistar/Desktop/JetRacer/track_inner1.log',offset_x,offset_y)
track_inner2 = load_map('/home/pistar/Desktop/JetRacer/track_inner2.log',offset_x,offset_y)

bounds_map = Polygon(coords)
track = Polygon(track_outer, holes=[track_inner1,track_inner2])
track = fix_map_rotation(track,coords,offset_x,offset_y)
attributes = {"id": 1, "name": "track", "valid": True}
centerline = Centerline(track, **attributes).geometry

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
def receive_image(conn):
    data = conn.recv(INT_BYTE_LIMIT)
    size = int.from_bytes(data, "big")
    data = bytearray()
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            return None
        data.extend(packet)
    image = np.frombuffer(data, dtype=np.uint8)
    logging.debug('Image shape ' + str(image.shape))
    #image = np.reshape(image, (128, 96, 1))
    image = cv2.imdecode(image, cv2.COLOR_BGR2GRAY)
    return image

def receive_battery(conn):
    batt_lvl = struct.unpack("!f", conn.recv(INT_BYTE_LIMIT))[0]
    return batt_lvl

@log
def send_action(conn, action):
    for act in action:
        packed = struct.pack("!f", act)
        conn.sendall(packed)


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in degrees (counterclockwise)
    pitch is rotation around y in degrees (counterclockwise)
    yaw is rotation around z in degrees (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x*180/math.pi, pitch_y*180/math.pi, yaw_z*180/math.pi # in degrees


def point_reaching_distance_reward(state):
    x = state[0]
    y = state[1]
    dist = np.sqrt((center_coord[0]-x)**2 + (center_coord[1]-y)**2)
    return 1-(dist)

def point_at_centerline_reward(state):
    x = state[0]
    y = state[1]
    throttle = state[2]
    current_xy = Point(x,y)

    if current_xy.within(track):
        dist = current_xy.distance(centerline)
        return (1-(dist*2))
    else:
        dist = current_xy.distance(centerline)
        if -dist>-1:
            return -dist
        else: return -.99


class ModelCar(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ModelCar, self).__init__()
        self.discrete = False
        self.use_vr = True
        self.act_interval = 500      # milliseconds
        self.state = None
        self.rew_fn = point_at_centerline_reward
        self.steering = 0.0
        self.throttle = 0.0
        self.total_reward = 0.
        self.vae = torch.load('/home/pistar/Desktop/JetRacer/dataset/dataset/vae.pth')
        self.log_file = open("deepracer_model_logs.csv", "a")
        self.episd_rw_file = open("/home/pistar/Desktop/JetRacer/deepracer_epis_rw_logs.csv", "a")
        self.total_steps = 5600

        if self.discrete:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(
                np.array([-1., -1.]).astype(np.float32),
                np.array([1., 1.]).astype(np.float32),
            )

        # self.observation_space = spaces.Box(
        #     low=0, high=255, shape=(CAMERA_WIDTH, CAMERA_HEIGHT, 1), dtype=np.uint8
        # )
        self.observation_space = spaces.Box(
            low=-100, high=100, shape=(32,), dtype=float
        )

        # Connect to car
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((SERVER_IP, PORT))
        s.listen()
        print('Waiting for client connection...')	
        self.conn, addr = s.accept()
        atexit.register(self.conn.close)
        print(f"Connected by {addr}")

        if self.use_vr:
            print('Connecting to VR sensor...')
            self.vr_sensor = connect("ws://localhost:8080/ws")        
            self.keep_going = True
            def keep_alive():
                while self.keep_going:
                    self.vr_sensor.recv()
            self.keep_alive_fn = keep_alive
            self.pool = ThreadPoolExecutor(max_workers=5)
            self.keep_thr = self.pool.submit(self.keep_alive_fn)
            atexit.register(self.vr_sensor.close)
            def stop_going():
                self.keep_going = False
            atexit.register(stop_going)
        print('Ready')

        self.num_steps = 0
        self.battery_level = None
        self.ep_steps = 0
        self.episode = 1
        self.ep_reward = []


    @log
    def get_pose(self):
        if self.use_vr == False:
            return [-1]*5
        self.keep_going = False
        self.keep_thr.result()
        while True:
            message = self.vr_sensor.recv()
            # If statement to grab specific lines received from the websocket (small filter)
            if ' KN0 POSE ' in message: # KNO POSE is used to grab coordinates
            	break
        data = message.split(' ')
        data2 = [float(data[each]) for each in range(3,10) if data[each]] 
        x, y, z, *other= data2
        a, b, c, d, *other = other
        roll, pitch, yaw = euler_from_quaternion(a,b,c,d)

        self.keep_going = True
        self.keep_thr = self.pool.submit(self.keep_alive_fn)
        return x, y, roll, pitch, yaw

    
    # ONLY USE get_state on initial reset(), always use do_action otherwise
    @log
    def get_state(self):
        # TODO include steering and throttle in state
        image = receive_image(self.conn)
        image = this_transform(image)
        image = np.expand_dims(image, axis=0)
        image = torch.Tensor(image)
        #print(image.shape)
        image = image.cuda()
        image, _, _ = self.vae.encode(image)
        image = image.cpu().detach()
        self.battery_level = 5#receive_battery(self.conn)
        # TODO return image
        # Don't use the image, use coordinates only to start with
        x, y, roll, pitch, yaw = self.get_pose()
        return image#np.asarray([x, y, roll, pitch, yaw])


    @log
    def do_action(self, act):
        send_action(self.conn, (act[0],0.8))
        #if act[1]>=0: send_action(self.conn, act)
        #else: send_action(self.conn, (act[0],act[1]))
        return self.get_state()
    

    def step(self, action):
        self.ep_steps += 1
        self.num_steps += 1
        terminated = False
        truncated = False

        steering, throttle = self.convert_action(action)

        if not self.use_vr:
            in_bounds = True
            in_track = True
        else:
            x, y, _, _, _ = self.get_pose()
            current_xy = Point(x, y)
            in_track = current_xy.within(track)
            in_bounds = current_xy.within(bounds_map)
            #print(np.round(x, 2), np.round(y, 2))

        #if in_bounds == False:
        if in_track == False:
            step_reward = -1
            terminated = True
        else:
            self.state = self.do_action((steering, throttle))
            step_reward = self.rew_fn([x, y, throttle])
            
        if terminated == False:
            offset = self.act_interval - (int(time.time() * 1000) % self.act_interval)
            time.sleep(offset/1000)

        self.total_reward += step_reward
        output = 'Step:' + str(self.num_steps+self.total_steps) + ',Battery:' + str(self.battery_level) + ',Reward:' + str(np.round(step_reward, 2)) + ',in_bounds:' + str(in_bounds) + ',in_track:' + str(in_track) + ',dist:' + str(np.round(current_xy.distance(centerline),2)) + ',steering:' + str(np.round(steering, 2)) + ',throttle: ' + str(np.round(throttle, 2)) + ',total_reward:' + str(np.round(self.total_reward, 2))+'\n'
        print(output,end='')

        log_file = open("deepracer_model_logs.csv", "a")
        log_file.write(output)
        log_file.close()

        return self.state, step_reward, terminated, truncated, {}
    
    def episode_printout(self):
        if len(self.ep_reward) <= 1:
            print('#--------------------#',sep='')
            print('# Episode:          ',self.episode,sep='')
            print('# Episode Steps:    ',self.ep_steps,sep='')
            print('# Total Reward:     ',np.round(self.total_reward,2),sep='')
            print('# Avg Reward/Step:  ',0,sep='')
            print('# Episode % Change: ',0,sep='')
            print('#------------------------------#',sep='')       
        else:
            print('#------------------------------#',sep='')
            print('# Episode:          ',self.episode,sep='')
            print('# Episode Steps:    ',self.ep_steps,sep='')
            print('# Total Reward:     ',np.round(self.total_reward,2),sep='')
            print('# Avg Reward/Step:  ',np.round(self.total_reward/self.ep_steps,2),sep='')
            print('# Avg Reward/Epis:  ',np.round(sum(self.ep_reward)/len(self.ep_reward),2))
            if self.ep_reward[-2] != 0: 
                print('# Episode % Change: ',np.round(((self.ep_reward[-1]-self.ep_reward[-2])/self.ep_reward[-2])*100,2),'%',sep='')
            else: print('# Episode % Change: 0 Division Error',sep='')
            print('#------------------------------#',sep='')


    def reset(self, **kwargs):
        episd_rw_file = open("/home/pistar/Desktop/JetRacer/deepracer_epis_rw_logs.csv", "a")
        episd_rw_file.write(str(self.episode) + ',' + str(self.total_reward) + '\n')
        episd_rw_file.close()

        self.episode_printout()
        self.ep_reward.append(self.total_reward)
        self.episode+=1
        self.ep_steps = 0
        self.total_reward = 0.0
        if self.state is None: 
            self.state = self.get_state()
        else:   # Reset vehicle with static policy to move back inside bounds
            x, y, _, _, _ = self.get_pose()
            current_x_y = Point(x,y)
            while not current_x_y.within(track):
                self.do_action((0.0, 0.0))
                time.sleep(0.5)
                print('Out of bounds')

                x, y, _, _, _ = self.get_pose()
                current_x_y = Point(x,y)
            '''
            while not current_x_y.within(bounds_map): 
                x, y, roll, pitch, yaw = self.get_pose()
                roll = roll%360
                newX = x-center_coord[0]
                newY = y-center_coord[1]
                angle = -(math.degrees(math.atan2(newY,newX)) + 90 ) % 360
                current_x_y = Point(x,y)

                max_throttle = 1.0
                if angle - 90 < roll and roll < angle +90:
                    if not (angle-20 < roll and roll < angle+20):
                        print('forward not facing ',end='')
                        if roll > angle:
                            print('turning left')
                            self.do_action((0.99, max_throttle))
                        else:
                            print('turning right')
                            self.do_action((-0.99, max_throttle))
                    else:
                        print('facing')
                        self.do_action((0,max_throttle))
                else:
                    roll = roll%180
                    if not (angle-10 < roll and roll < angle+10):
                        print('backward not facing ',end='')
                        if roll < angle:
                            print('turning left')
                            self.do_action((0.99, -max_throttle))
                        else:
                            print('turning right')
                            self.do_action((-0.99, -max_throttle))
                    else:
                        print('facing')
                        self.do_action((0.,-max_throttle))
                time.sleep(0.5)
            '''
            '''
                if ((angle-10) < roll) and (roll < (angle+10)):
                    print('facing')
                else:
                    print('not facing')
                '''

                #print('angle:', angle, 'roll:', roll, 'x:', x, 'y:', y)
                #print('Out of bounds')
            '''
                self.state = self.do_action((0.0, 0.0))
                self.steering = 0.0
                self.throttle = 0.0
                time.sleep(3)
                '''
        print('in bounds...')
        time.sleep(3)
        return self.state, {}
    
    
    def convert_action(self, action):
        if self.discrete:
            self.throttle = 0.6
            if action == 0:   # turn left more
                if self.steering < 1:
                    self.steering = self.steering + STEERING_CHANGE
            elif action == 1:   # turn right more
                if self.steering > -1:
                    self.steering = self.steering - STEERING_CHANGE
            elif action == 2:
                self.throttle = 0.0
            elif action == 3:
                self.steering = 0.0

            # if action == 0:     # noop
            #     pass
            # elif action == 1:   # increase speed
            #     if self.throttle < 1:
            #         self.throttle = self.throttle + THROTTLE_CHANGE
            # elif action == 2:   # decrease speed
            #     if self.throttle > -1:
            #         self.throttle = self.throttle - THROTTLE_CHANGE
            # elif action == 3:   # turn left more
            #     if self.steering < 1:
            #         self.steering = self.steering + STEERING_CHANGE
            # elif action == 4:   # turn right more
            #     if self.steering > -1:
            #         self.steering = self.steering - STEERING_CHANGE
            # else:
            #     raise NotImplementedError('Action undefined')
        # else:
        #         raise NotImplementedError('Action undefined')
        self.steering = action[0]
        self.throttle = action[1]
        return self.steering, self.throttle


    def render(self, mode='human', close=False):
        pass
