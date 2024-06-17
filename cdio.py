#!/usr/bin/env python3
import sys
import math
import time
import threading
from ev3dev2.motor import LargeMotor, MediumMotor, OUTPUT_A, OUTPUT_D, OUTPUT_C, SpeedPercent, MoveTank
from ev3dev2.sound import Sound

left_motor = LargeMotor(OUTPUT_D) 
right_motor = LargeMotor(OUTPUT_A) 
brush_motor = MediumMotor(OUTPUT_C) 
tank_drive = MoveTank(OUTPUT_D, OUTPUT_A)
sound = Sound()
movementSpeed = 70
turnSpeed = 50
wheel_diameter = 70  # mm
wheel_circumference = math.pi * wheel_diameter
wheel_distance = 170  # mm
robot_circumference = math.pi * wheel_distance

orders = []  

def turn(speed, angle):
        angle = angle*-0.45
        turn_circumference = (angle / (360)) * robot_circumference
        rotations = turn_circumference / wheel_circumference
        degrees_to_turn = rotations * 360
        tank_drive.on_for_degrees(SpeedPercent(-speed), SpeedPercent(speed), degrees_to_turn)
            
def turn_by_degrees(speed, degrees):
    tank_drive.on_for_degrees(SpeedPercent(-int(speed)), SpeedPercent(int(speed)), int(degrees))


def forward_by_degrees(speed, degrees):
    tank_drive.on_for_degrees(SpeedPercent(int(speed)), SpeedPercent(int(speed)), int(degrees))


def forward(speed, distance):
        rotations = distance / wheel_circumference
        degrees_to_turn = rotations * 360
        tank_drive.on_for_degrees(SpeedPercent(speed), SpeedPercent(speed), degrees_to_turn)

def stop_all_motors():
    orders.clear() 
    tank_drive.stop() 
    brush_motor.stop()
    sound.beep()  
    

def music():
    sound.play_song((
    ('A3', 'e3'),
    ('D3', 'q'),
    ('E3', 'e3'),
    ('F3', 's'),
    ('G3', 's'),
    ('F3', 'h'),
    ('A3', 'e3'),
    ('A3', 's'),
    ('D3', 'q'),
    ('E3', 's'),
    ('F3', 's'),
    ('A3', 's'),
    ('D3', 's'),
    ('A4', 's'),
    ('G3', 'h'),
    ('A3', 'e3'),
    ('D3', 'q'),
    ('E3', 's'),
    ('F3', 's'),
    ('D3', 's'),
    ('A4', 's'),
    ('F3', 's'),
    ('D4', 'w'),
    ('D3', 's'),
    ('F3', 's'),
    ('E3', 's'),
    ('D3', 's'),
    ('A4', 'e3'),
    ('F3', 's'),
    ('D3', 's'),
    ('A3', 'e3'),
    ('A3', 's'),
    ('A3', 's'),
    ('D3', 'w'),
    ('A4', 'e3'),
    ('D4', 'q'),
    ('E4', 'e3'),
    ('F4', 's'),
    ('G4', 's'),
    ('F4', 'h'),
    ('A4', 'q'),
    ('A4', 's'),
    ('D4', 'q'),
    ('E4', 's'),
    ('F4', 's'),
    ('A4', 's'),
    ('D4', 's'),
    ('A5', 's'),
    ('G4', 'w'),
    ('A4', 'e3'),
    ('D4', 'e3'),
    ('E4', 's'),
    ('F4', 's'),
    ('D4', 's'),
    ('A5', 's'),
    ('F4', 's'),
    ('D5', 'w'),
    ('D4', 's'),
    ('F4', 's'),
    ('E4', 's'),
    ('D4', 's'),
    ('A5', 'e3'),
    ('F4', 's'),
    ('D4', 's'),
    ('A4', 'e3'),
    ('A4', 's'),
    ('A4', 's'),
    ('D4', 'w')
))
    
def moveConstant(speed):
    if speed == 0:
        tank_drive.on_for_seconds(1, 1, 0)
    else: 
        tank_drive.on(SpeedPercent(speed), SpeedPercent(speed))
    
def turnConstant(speed):
    if speed == 0:
        tank_drive.on_for_seconds(1, 1, 0)
    else: 
        tank_drive.on(SpeedPercent(speed), SpeedPercent(-speed))

def brush(speed):
    if speed == 0:
        brush_motor.on_for_seconds(1, 0)
    else:
        brush_motor.on(SpeedPercent(speed))

def execute_order(order):
    if order[0] == "move":
        forward(movementSpeed, order[1])
    elif order[0] == "turn":
        turn(turnSpeed, order[1])
    elif order[0] == "start" or order[0] == "end":
        sound.beep()
    elif order[0] == "music":
        music()
    elif order[0] == "turn_degrees":
        turn_by_degrees(order[2], order[1])
    elif order[0] == "forward_degrees":
        forward_by_degrees(order[2], order[1])
    time.sleep(0.1)  # delay between orders

def check_orders_for_stop():
    for order in orders:
        if order[0] == "stop":
            stop_all_motors()
        elif order[0] == "clear":
            clear_queue()

def add_order(type, a, b):
    orders.append((type, a, b))

def listen_for_orders():
    print("____________\n\nReady for orders:\n\tmove <length> \n\tturn <angle>\n\tbrush <speed>\n\tcMove <speed> \n\tcTurn <speed>\n\tend\n\tforward_degrees <degrees> <speed>\n\tturn_degrees <degrees> <speed>\n____________")
    execute_order(('start', 0, 0))
    
    try:
        while True:
            check_orders_for_stop()
            if orders:
                execute_order(orders.pop(0))
                print("True")
            
            line = input()
            if not line:
                continue  
            
            parts = line.strip().split()
            if parts[0] == "music":
                music()
            elif parts[0] == "end":
                print("Ending program.")
                stop_all_motors()
                break  
            elif parts[0] == "turn":
                add_order(parts[0], int(parts[1]), None)
            elif parts[0] == "move":
                add_order(parts[0], int(parts[1]), None)
            elif parts[0] == "forward_degrees":
                add_order(parts[0], parts[1], parts[2])
            elif parts[0] == "turn_degrees":
                add_order(parts[0], parts[1], parts[2])
            elif parts[0] == "brush":
                brush(int(parts[1]))
                print("True")
            elif parts[0] == "cMove":
                moveConstant(int(parts[1]))
                print("True")
            elif parts[0] == "cTurn":
                turnConstant(int(parts[1]))
                print("True")
        stop_all_motors()
    except KeyboardInterrupt:
        print("Program interrupted by user.")


if __name__ == "__main__":
    print("True")
    listen_for_orders()
