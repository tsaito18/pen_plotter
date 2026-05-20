# coding=utf-8
'''
drawcore_serial.py
Serial connection utilities for DrawCore

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
'''


import logging
from packaging.version import parse
import time

from .plot_utils_import import from_dependency_import
inkex = from_dependency_import('ink_extensions.inkex')
serial = from_dependency_import('serial')
from serial.tools.list_ports import comports
text_utils = from_dependency_import('drawcore_plotink.text_utils')

logger = logging.getLogger(__name__)

def version():      # Version number for this document
    return "2.20"   # Dated 2025-06-29


def findPort():
    # Find first available Board by searching USB ports.
    # Return serial port object.
    try:
        from serial.tools.list_ports import comports
    except ImportError:
        return None
    if comports:
        try:
            com_ports_list = list(comports())
        except TypeError:
            return None
        drawcore_port = None
        for port in com_ports_list:
            if port[2].startswith("USB VID:PID=1A86:7523"):
                drawcore_port = port[0]  # Success; DrawCore found by VID/PID match.
                break
            if port[2].startswith("USB VID:PID=1A86:8040"):
                drawcore_port = port[0]  # Success; DrawCore found by VID/PID match.
                break
        if drawcore_port == None:
            for port in com_ports_list:
                logger.error('com_port: {0}'.format(port))
        return drawcore_port

def find_named_drawcore(port_name):
    '''
    Names should be 3-16 characters long. Comparisons are not case sensitive.
    (Name tags may assigned with the ST command on firmware 2.13 and later.)
    If found:     Return serial port name (enumeration)
    If not found, Return None
    '''
    if port_name is not None:
        name_upper = port_name.upper()
        drawcore_ports_list = listDRAWCOREports()
        if not drawcore_ports_list:
            return None
        for port in drawcore_ports_list:
            if name_upper == port[0].upper():
                return port[0]  # Success found by port match.
            try:
                serial_port = serial.Serial()
                serial_port.port = port[0]
                serial_port.baudrate = 115200
                serial_port.timeout = 1
                serial_port.rts = False
                serial_port.dtr = False
                serial_port.open()
                time.sleep(0.01)
                serial_port.write('$B\r'.encode('ascii'))
                re = serial_port.readline()#.decode('ascii')
                re = serial_port.readline()#.decode('ascii')
                serial_port.reset_input_buffer()
                serial_port.write('v\r'.encode('ascii'))
                str_version = serial_port.readline()
                if not str_version.startswith("DrawCore".encode('ascii')):
                    continue
                name = query(serial_port, '$QT\r')
                name = name.strip()
                if name == name_upper:
                    serial_port.close()
                    return port[0]
                serial_port.close() 
            except serial.SerialException as err:
                # logger.error("find_named_drawcore:\n Error testing serial port `{}` connection".format(port[0]))
                pass
    return None


def find_named_drawcore_then_testPort(port_name):
    '''
    Names should be 3-16 characters long. Comparisons are not case sensitive.
    (Name tags may assigned with the ST command on firmware 2.13 and later.)
    If found:     Return serial port name (enumeration)
    If not found, Return None
    '''
    if port_name is not None:
        name_upper = port_name.upper()

        drawcore_ports_list = listDRAWCOREports()
        if not drawcore_ports_list:
            return None
        for port in drawcore_ports_list:
            try:
                # serial_port = serial.Serial(port_name, timeout=1.0)  # 1 second timeout!
                serial_port = serial.Serial()
                serial_port.port = port[0]
                serial_port.baudrate = 115200
                serial_port.timeout = 1
                serial_port.rts = False
                serial_port.dtr = False
                serial_port.open()
                serial_port.write('$B\r'.encode('ascii'))
                re = serial_port.readline()#.decode('ascii')
                re = serial_port.readline()#.decode('ascii')
                serial_port.reset_input_buffer()
                serial_port.write('v\r'.encode('ascii'))
                str_version = serial_port.readline()
                if not str_version.startswith("DrawCore".encode('ascii')):
                    continue
                name = query(serial_port, '$QT\r')
                name = name.strip()
                if name == name_upper:
                    return serial_port
                if name_upper == port[0].upper():
                    return serial_port # Success found by port match.
                serial_port.close() 
            except serial.SerialException as err:
                logger.error("find_named_drawcore_then_testPort:\n Error testing serial port `{}`".format(port[0]))
                # pass
    return None


def query_nickname(port_name, verbose=True):
    '''
    Query the DrawCore nickname and report it.
    If verbose is True or omitted, the result will be human readable.
    A short version is returned if verbose is False.
    '''
    if port_name is not None:
        version_status = min_version(port_name, "2.13")

        if version_status:
            raw_string = query(port_name, '$QT\r')
            if raw_string.isspace():
                if verbose:
                    return "This iDraw does not have a nickname assigned."
                return None
            if verbose:
                return "iDraw nickname: " + raw_string
            return str(raw_string).strip()
        if version_status is False:
            if verbose:
                return "iDraw naming requires firmware version 2.13 or higher."
    return None


def write_nickname(port_name, nickname):
    '''
    Write the DrawCore nickname.
    '''
    if port_name is not None:
        version_status = min_version(port_name, "2.13")

        if version_status:
            try:
                cmd = '$ST=' + nickname + '\r'
                command(port_name,cmd)
                return True
            except:
                return False
    return None
   
def list_port_info():
    '''Find and return a list of all USB devices and their information.'''
    try:
        com_ports_list = list(comports())
    except TypeError:
        return None

    port_info_list = []
    for port in com_ports_list:
        port_info_list.append(port[0]) # port name
        port_info_list.append(port[1]) # Identifier
        port_info_list.append(port[2]) # VID/PID
    if port_info_list:
        return port_info_list
    return None    

def listDRAWCOREports():
    # Find and return a list of all DrawCore units
    # connected via USB port.
    try:
        from serial.tools.list_ports import comports
    except ImportError:
        return None
    if comports:
        com_ports_list = list(comports())
        drawcore_ports_list = []
        for port in com_ports_list:
            port_has_drawcore = False
            if port[2].startswith("USB VID:PID=1A86:7523"):
                port_has_drawcore = True  # Success; DrawCore found by VID/PID match.
            if port[2].startswith("USB VID:PID=1A86:8040"):
                port_has_drawcore = True  # Success; DrawCore found by VID/PID match.
            if port_has_drawcore:
                drawcore_ports_list.append(port)
        if drawcore_ports_list:
            return drawcore_ports_list

def list_named_drawcores():
    '''Return descriptive list of all DrawCore units'''
    drawcore_ports_list = listDRAWCOREports()
    if not drawcore_ports_list:
        return None
    drawcore_names_list = []
    for port in drawcore_ports_list:
        name_found = False
        try:
            serial_port = serial.Serial()
            serial_port.port = port[0]
            serial_port.baudrate = 115200
            serial_port.timeout = 1
            serial_port.rts = False
            serial_port.dtr = False
            serial_port.open()
            time.sleep(0.01)
            serial_port.write('$B\r'.encode('ascii'))
            re = serial_port.readline()#.decode('ascii')
            re = serial_port.readline()#.decode('ascii')
            serial_port.reset_input_buffer()
            serial_port.write('v\r'.encode('ascii'))
            str_version = serial_port.readline()
            if not str_version.startswith("DrawCore".encode('ascii')):
                continue
            name = query(serial_port, '$QT\r')
            name = name.strip()
            if len(name) < 3:
                name = None
            if name is not None:
                drawcore_names_list.append(name)
                name_found = True
            serial_port.close() 
        except serial.SerialException as err:
            logger.error("list_named_drawcores:\nError testing serial port `{}` connection".format(port[0]))
        if not name_found:
            drawcore_names_list.append(port[0])    
    return drawcore_names_list
                        
def testPort(port_name):
    """
    Open a given serial port, verify that it is an DrawCore,
    and return a SerialPort object that we can reference later.

    This routine only opens the port;
    it will need to be closed as well, for example with closePort( port_name ).
    You, who open the port, are responsible for closing it as well.
    """
    if port_name is not None:
        try:
            # serial_port = serial.Serial(port_name, timeout=1.0)  # 1 second timeout!
            serial_port = serial.Serial()
            serial_port.port = port_name
            serial_port.baudrate = 115200
            serial_port.timeout = 1
            serial_port.rts = False
            serial_port.dtr = False
            serial_port.open()
            serial_port.write('$B\r'.encode('ascii'))
            re = serial_port.readline()
            re = serial_port.readline()
            serial_port.reset_input_buffer()
            serial_port.write('v\r'.encode('ascii'))
            str_version = serial_port.readline()
            if str_version and str_version.startswith("DrawCore".encode('ascii')):
                return serial_port   
            serial_port.reset_input_buffer()
            serial_port.write('v\r'.encode('ascii'))
            str_version = serial_port.readline()
            if str_version and str_version.startswith("DrawCore".encode('ascii')):
                return serial_port
            serial_port.close()
        except serial.SerialException as err:
            logger.error("Error testing serial port `{}` connection".format(port_name))
            logger.info("Error context:", exc_info=err)
        return None


def openPort():
    # Find and open a port to a single attached DrawCore.
    # The first port located will be used.
    found_port = findPort()
    serial_port = testPort(found_port)
    if serial_port:
        return serial_port
    else:
        logger.error('error  open com_port: {0}'.format(found_port))
        return None


def closePort(port_name):
    if port_name is not None:
        try:
            port_name.close()
        except serial.SerialException:
            pass


def query(port_name, cmd, verbose=True):
    if port_name is not None and cmd is not None:
        response = ''
        try:
            port_name.write(cmd.encode('ascii'))
            response = port_name.readline().decode('ascii')
            n_retry_count = 0
            while len(response) == 0 and n_retry_count < 10:
                # get new response to replace null response if necessary
                response = port_name.readline()
                n_retry_count += 1
            if cmd.split(",")[0].strip().lower() not in ["v", "i", "a", "mr", "pi", "qm"]:
                # Most queries return an "OK" after the data requested.
                # We skip this for those few queries that do not return an extra line.
                unused_response = port_name.readline()  # read in extra blank/OK line
                n_retry_count = 0
                while len(unused_response) == 0 and n_retry_count < 10:
                    # get new response to replace null response if necessary
                    unused_response = port_name.readline()
                    n_retry_count += 1
        except (serial.SerialException, IOError, RuntimeError, OSError) as err:
            if verbose:
                logger.error("Error reading serial data")
            else:
                logger.info("Error reading serial data")
            logger.info("Error context:", exc_info=err)
        return response
    return None

def query_all(port_name, cmd):
    if port_name is not None and cmd is not None:
        response = ''
        try:
            port_name.write(cmd.encode('ascii'))
            time.sleep(1)
            response = port_name.read_all().decode('ascii')
            response = response.replace('\r','')
        except (serial.SerialException, IOError, RuntimeError, OSError) as err:
            logger.error("Error reading serial data")
            logger.info("Error context:", exc_info=err)
        return response
    return None


def command(port_name, cmd, verbose=True):
    if port_name is not None and cmd is not None:
        try:
            port_name.write(cmd.encode('ascii'))
            response = port_name.readline().decode('ascii')
            n_retry_count = 0
            while len(response) == 0 and n_retry_count < 100:
                # get new response to replace null response if necessary
                response = port_name.readline().decode('ascii')
                n_retry_count += 1
            if response.strip().startswith("ok"):
                # Debug option: indicate which command:
                # inkex.errormsg( 'OK after command: ' + cmd )
                pass
            else:
                if response:
                    error_msg = '\n'.join(('Unexpected response from DrawCore.',
                                           '    Command: {0}'.format(cmd.strip()),
                                           '    Response: {0}'.format(response.strip())))
                else:
                    error_msg = 'DrawCore Serial Timeout after command: {0}'.format(cmd)
                if verbose:
                    logger.error(error_msg)
                else:
                    logger.info(error_msg)
        except (serial.SerialException, IOError, RuntimeError, OSError) as err:
            if cmd.strip().lower() not in ["rb"]: # Ignore error on reboot (RB) command
                if verbose:
                    logger.error('Failed after command: {0}'.format(cmd))
                else:
                    logger.info('Failed after command: {0}'.format(cmd))
                logger.info("Error context:", exc_info=err)

def min_version(port_name, version_string):
    # Query the DrawCore firmware version for the DrawCore located at port_name.
    # Return True if the DrawCore firmware version is at least version_string.
    # Return False if the DrawCore firmware version is below version_string.
    # Return None if we are unable to determine True or False.

    if port_name is not None:
        drawcore_version_string = queryVersion(port_name)  # Full string, human readable
        drawcore_version_string = drawcore_version_string.split("DrawCore V", 1)
        if len(drawcore_version_string) > 1:
            drawcore_version_string = drawcore_version_string[1]
            drawcore_version_string = drawcore_version_string[:4]
        else:
            return None  # We haven't received a reasonable version number response.

        drawcore_version_string = drawcore_version_string.strip()  # Stripped copy, for number comparisons
        if parse(drawcore_version_string) >= parse(version_string):
            return True
        return False
    return None


def queryVersion(port_name):
    return query(port_name, 'V\r', True)  # Query DrawCore Version String
