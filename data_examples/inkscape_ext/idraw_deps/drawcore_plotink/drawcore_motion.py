# coding=utf-8
# drawcore_motion.py

"""
Motion control utilities for DrawCore

The MIT License (MIT)

Copyright (c) 2025 idraw team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from . import drawcore_serial


def version():  # Report version number for this document
    ''' Return version number '''
    return "2.20"   # Dated 2025-06-29



def doTimedPause(port_name, n_pause, verbose=True):
    ''' "Hardware" pause on EBB control board '''
    if port_name is not None:
        pass
        while n_pause > 0:
            if n_pause > 750:
                time_delay = 750
            else:
                time_delay = n_pause
                if time_delay < 1:
                    time_delay = 1  # don't allow zero-time moves
            # drawcore_serial.command(port_name, 'SM,{0},0,0\r'.format(time_delay))
            # drawcore_serial.command(port_name, 'G4 P{0}\r'.format(time_delay/1000.0))
            n_pause -= time_delay

def doXYMove(port_name, delta_x, delta_y, duration, verbose=True):
    '''
    Move X/Y axes as: "SM,<move_duration>,<axis1>,<axis2><CR>"
    Typically, this is wired up such that axis 1 is the Y axis and axis 2 is the X axis of motion.
    On EggBot, Axis 1 is the "pen" motor, and Axis 2 is the "egg" motor.
    '''
    XSteps = -(delta_x + delta_y)/2.0
    YSteps = (delta_x - delta_y)/2.0
    if port_name is not None:
        # pass
        str_output = 'G1G91X{0}Y{1}\r'.format(YSteps/100, XSteps/100)
        drawcore_serial.command(port_name, str_output)
        
def doMove(port_name, delta_x, delta_y, speed):
    '''
    
    '''
    if port_name is not None:
        # pass
        str_output = 'G1G91X{0}Y{1}F{2}\r'.format(delta_x, delta_y,speed)
        drawcore_serial.command(port_name, str_output)

def QueryPenUp(port_name, verbose=True):
    """ Check if the pen is up, using QP. """
    if port_name is not None:
        pen_status = drawcore_serial.query(port_name, '$QP\r', verbose)
        if pen_status[0] == '0':
            return False
        return True


def QueryPRGButton(port_name, verbose=True):
    """ Check if the button has been pressed, using QB. """
    if port_name is not None:
        return drawcore_serial.query(port_name, '$B\r', verbose)
    return None

def sendDisableMotors(port_name, verbose=True):
    """ Disable stepper motors with EM command """
    if port_name is not None:
        drawcore_serial.command(port_name, '$SLP\r', verbose)
        pass


def sendEnableMotors(port_name, res, verbose=True):
    """
    Enable both motors with EM command at selected resolution.
        If res == 0, -> Motor disabled
        If res == 1, -> 16X microstepping
        If res == 2, -> 8X microstepping
        If res == 3, -> 4X microstepping
        If res == 4, -> 2X microstepping
        If res == 5, -> No microstepping
    """
    if res < 0:
        res = 0
    if res > 5:
        res = 5
    # if port_name is not None:
        # drawcore_serial.command(port_name, 'EM,{0},{0}\r'.format(res))

def query_enable_motors(port_name, verbose=True):
    return None, None


def sendPenDown(port_name, z_pos,speed,speed_pendown, verbose=True):
    if port_name is not None:
        str_output = 'G1G90 Z{0}F{1}\r'.format(z_pos, speed)
        drawcore_serial.command(port_name, str_output, verbose)
        drawcore_serial.command( port_name, 'G1 F{0}\r'.format(speed_pendown), verbose)


def sendPenUp(port_name, z_pos,speed,speed_penup, verbose=True):
    if port_name is not None:
        str_output = 'G1G90 Z{0}F{1}\r'.format(z_pos, speed)
        drawcore_serial.command(port_name, str_output, verbose)
        drawcore_serial.command( port_name, 'G1 F{0}\r'.format(speed_penup), verbose)

def PBOutConfig(port_name, pin, state, verbose=True):
    """
    Enable an I/O pin. Pin: {0,1,2, or 3}. State: {0 or 1}.
    Note that B0 is used as an alternate pause button input.
    Note that B1 is used as the pen-lift servo motor output.
    Note that B3 is used as the EggBot engraver output.
    For use with a laser (or similar implement), pin 3 is recommended
    """
    if port_name is not None:
        pass


def PBOutValue(port_name, speed, state, verbose=True):
    """
    Set state of the I/O pin. Pin: {0,1,2, or 3}. State: {0 or 1}.
    Set the pin as an output with OutputPinBConfigure before using this.
    """
    if port_name is not None:
        str_output = 'G1 F{0} M3 S{1}\r'.format(speed, state)
        drawcore_serial.command(port_name, str_output, verbose)
        # pass
        # str_output = 'PO,B,{0},{1}\r'.format(pin, state)
        # drawcore_serial.command(port_name, str_output)


def TogglePen(port_name, pen_pos_up, pen_pos_down, speed):
    """ Toggle pen state using TP """
    if port_name is not None:
        drawcore_serial.command( port_name, 'G90 G1 F{0}\r'.format(speed))
        str_output = '$TP{0:0.1f},{1:0.1f}\r'.format(pen_pos_up, pen_pos_down, verbose)
        drawcore_serial.command(port_name, str_output, verbose)

def GoHome(port_name):
    if port_name is not None:
        drawcore_serial.command(port_name, '$H\r')

def setPenDownPos(port_name, servo_max, verbose=True):
    """ Set pen down position using SC """
    if port_name is not None:
        pass#drawcore_serial.command(port_name, 'SC,5,{0}\r'.format(servo_max))
        # servo_max may be in the range 1 to 65535, in units of 83 ns intervals.
        # This sets the "Pen Down"position.



def setPenDownRate(port_name, pen_down_rate):
    """ Set pen lowering speed using SC """
    if port_name is not None:
        pass#drawcore_serial.command(port_name, 'SC,12,{0}\r'.format(pen_down_rate))
        # Set the rate of change of the servo when going down.



def setPenUpPos(port_name, servo_min, verbose=True):
    """ Set pen up position using SC """
    if port_name is not None:
        pass#drawcore_serial.command(port_name, 'SC,4,{0}\r'.format(servo_min))
        # servo_min may be in the range 1 to 65535, in units of 83 ns intervals.



def setPenUpRate(port_name, pen_up_rate, verbose=True):
    """ Set pen raising speed using SC """
    if port_name is not None:
        pass
        # drawcore_serial.command(port_name, 'SC,11,{0}\r'.format(pen_up_rate))



def setEBBLV(port_name, ebb_lv, verbose=True):
    """
    Set the EBB "Layer" Variable, an 8-bit number we can read and write.
    (Unrelated to document layers; name is an historical artifact.)
    """
    if port_name is not None:
        pass
        # drawcore_serial.command(port_name, 'SL,{0}\r'.format(ebb_lv))


def queryEBBLV(port_name, verbose=True):
    """
    Query the EBB "Layer" Variable, an 8-bit number we can read and write.
    (Unrelated to document layers; name is an historical artifact.)
    """
    # if port_name is not None:
        # value = drawcore_serial.query(port_name, 'QL\r')
        # try:
            # ret_val = int(value)
            # return ret_val
        # except:
            # return None
    return 0


def queryVoltage(port_name):

    return True


def servo_timeout(port_name, timeout_ms, state=None):
    """
    Set the EBB servo motor timeout.
    The EBB will cut power to the pen-lift servo motor after a given
    time delay since the last X/Y/Z motion command.
    It can also optionally set an immediate on/off state.

    The time delay timeout_ms is given in ms. A value of 0 will
    disable the automatic power-off feature.

    The state parameter is given as 0 or 1, to turn off or on
    servo power immediately, respectively.

    This feature requires EBB hardware v 2.5.0 and firmware 2.6.0

    Reference: http://evil-mad.github.io/EggBot/ebb.html#SR
    """
    if port_name is not None:
        pass
        # if not drawcore_serial.min_version(port_name, "2.6.0"):
            # return      # Unable to read version, or version is below 2.6.0.
        # if state is None:
            # str_output = 'SR,{0}\r'.format(timeout_ms)
        # else:
            # str_output = 'SR,{0},{1}\r'.format(timeout_ms, state)
        # drawcore_serial.command(port_name, str_output)
