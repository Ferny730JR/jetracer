import subprocess
import sys
import time
import re
import pynput
controller = subprocess.Popen(['sudo', 'jstest', '/dev/input/js0'],stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')

from pynput import keyboard
from pynput.keyboard import Key

def on_key_press(key):
    if key == Key.right:
        print("Right key press")
    elif key == Key.left:
        print("Left key press")
    elif key == Key.up:
        print("Up key press")
    elif key == Key.down:
        print("Down key press")
    elif key == Key.esc:
        exit()

def on_key_release(key):
    if key == Key.right:
        print("Right key released")
    elif key == Key.left:
        print("Left key released")
    elif key == Key.up:
        print("Up key released")
    elif key == Key.down:
        print("Down key released")
    elif key == Key.esc:
        exit()


with keyboard.Listener(on_press=on_key_press,on_release=on_key_release) as listener:
    listener.join()


#for line in iter(controller.stdout.readline, b''):
#    print(">>> " + line.rstrip())
'''
while True:
    realtime_output = controller.stdout.readline()

    if realtime_output == '' and controller.poll() is not None:
        print('breaking')
        break

    if realtime_output:
        out = realtime_output.strip()#, flush=True)
        #print(out)
        out = re.split(r':\s+|\s+|:',out)
        if len(out)>4:
            #print(out)
            print(out[2],out[4])
    #time.sleep(0.5)
'''


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

#run_command(['sudo', 'jstest', '/dev/input/js0'])