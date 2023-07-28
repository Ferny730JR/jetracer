from shapely.geometry import Point, MultiPoint, Polygon
from shapely import affinity
from shapely.ops import nearest_points
from centerline.geometry import Centerline

import math
import numpy as np
import asyncio
from websockets.sync.client import connect
import blissful_basics as bb
import time
import threading
from collections import deque

import matplotlib.pyplot as plt

# Load maps
def load_map(filename, offset_x, offset_y):
    track = []
    track_coords = open(filename, 'r')

    for coord in track_coords:
        t = coord.split(',')
        track.append((float(t[0])+offset_x,float(t[1])+offset_y))
    
    track_coords.close()
    return track

# Create a Polygon
def fix_map_rotation(track,bounds_map,offset_x,offset_y,error_threshold=1):
    bounds_map_calib=[(0.024254 , -1.776745),(0.58907 , 0.523119),(4.273964 , -0.46625),(3.576584 , -2.787911)] #DONT CHANGE
    coords = [(np.round(coord[0]+offset_x,5),np.round(coord[1]+offset_y,5)) for coord in bounds_map_calib]
    print(coords)
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

poly = Polygon(coords)
track = Polygon(track_outer, holes=[track_inner1,track_inner2])
track = fix_map_rotation(track,coords,offset_x,offset_y)

attributes = {"id": 1, "name": "track", "valid": True}
centerline = Centerline(track,**attributes).geometry


#l = centerline.geometry
#print(l.line_locate_point(pos),normalized=True)
'''
if current position is in track_outer and also outside both track_inner1 and 2
'''
#lt.plot(*l.xy)

#plt.show()

should_update = bb.countdown(50)
at_boundary = False

def euler_from_quaternion(x, y, z, w):
		"""
		Convert a quaternion into euler angles (roll, pitch, yaw)
		roll is rotation around x in radians (counterclockwise)
		pitch is rotation around y in radians (counterclockwise)
		yaw is rotation around z in radians (counterclockwise)
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

def centroid_poly(X, Y):
    """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
    N = len(X)
    # minimal sanity check
    if not (N == len(Y)): raise ValueError('X and Y must be same length.')
    elif N < 3: raise ValueError('At least 3 vertices must be passed.')
    sum_A, sum_Cx, sum_Cy = 0, 0, 0
    last_iteration = N-1
    # from 0 to N-1
    for i in range(N):
        if i != last_iteration:
            shoelace = X[i]*Y[i+1] - X[i+1]*Y[i]
            sum_A  += shoelace
            sum_Cx += (X[i] + X[i+1]) * shoelace
            sum_Cy += (Y[i] + Y[i+1]) * shoelace
        else:
            # N-1 case (last iteration): substitute i+1 -> 0
            shoelace = X[i]*Y[0] - X[0]*Y[i]
            sum_A  += shoelace
            sum_Cx += (X[i] + X[0]) * shoelace
            sum_Cy += (Y[i] + Y[0]) * shoelace
    A  = 0.5 * sum_A
    factor = 1 / (6*A)
    Cx = factor * sum_Cx
    Cy = factor * sum_Cy
    # returning abs of A is the only difference to
    # the algo from above link
    return Cx, Cy, abs(A)

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


def hello():
    global at_boundary #declare as a global var to use in another file
    # Connects to websocket (8080 for our simulation)
    with connect("ws://localhost:8080/ws") as websocket:
        while True:
            message = websocket.recv()
            # If statement to grab specific lines received from the websocket (small filter)
            if ' KN0 POSE ' in message: # KNO POSE is used to grab coordinates
                data = message.split(' ')
                data2 = [float(data[each]) for each in range(3,10) if data[each]] 
                x, y, z, *other=data2
                a,b,c,d = other
                a,b,c = euler_from_quaternion(a,b,c,d)
                a = a%360

                if should_update():
                    newX = x-center_coord[0]
                    newY = y-center_coord[1]

                    angle = -(math.degrees(math.atan2(newY,newX)) + 90 ) %360
                    #if ((angle-10) < a) and (a < (angle+10)):
                    #    print('facing')
                    #else:
                    #    print('not facing')

                    pos = Point(x, y)
                    at_boundary = pos.within(poly)
                    
                    dist = np.sqrt((center_coord[0]-x)**2 + (center_coord[1]-y)**2)
                    step_reward = 1-(dist)
                    
                    if pos.within(track): #and not pos.within(track_inner1) and not pos.within(track_inner2):
                         print('in track, ',end='')
                    else: print('outside track, ',end='')
                    print('dist:',np.round(pos.distance(centerline),2),end=' ')
                    print('x: ',np.round(x,2),', y: ',np.round(y,2),sep='',end=' ')
                    print('calc_ang:',np.round(angle,2), 'actual_ang:', np.round(a,2),end=' ')
                    print('in bounds:', at_boundary, ', reward:',np.round(point_at_centerline_reward((x,y,1)),2),sep='')
                    
                    #print(x,', ',y,sep='')

hello()
