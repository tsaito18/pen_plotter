#
# Copyright 2025 idraw team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""
serial_utils.py

This module modularizes some serial functions..

Requires Python 3.7 or newer.

"""

import time
from idraw2_0internal.plot_utils_import import from_dependency_import
from idraw2_0internal.idraw_options import versions as ad_versions
drawcore_serial = from_dependency_import('drawcore_plotink.drawcore_serial') 
drawcore_motion = from_dependency_import('drawcore_plotink.drawcore_motion')

def connect(options, plot_status, message_fun, logger):
    """ Connect to iDraw over USB """
    port_name = None
    if options.port_config == 1: # port_config value "1": Use first available iDraw.
        options.port = None
    if not options.port: # Try to connect to first available iDraw.
        plot_status.port = drawcore_serial.openPort()
    elif str(type(options.port)) in (
            "<type 'str'>", "<type 'unicode'>", "<class 'str'>"):
        # This function may be passed a port name to open (and later close).
        options.port = str(options.port).strip('\"')
        options.port = options.port.replace(" ", "")
        port_name = options.port
        plot_status.port = drawcore_serial.find_named_drawcore_then_testPort(options.port)
        options.port = None  # Clear this input, to ensure that we close the port later.
    else:
        # options.port may be a serial port object of type serial.serialposix.Serial.
        # In that case, interact with that given port object, and leave it open at the end.
        plot_status.port = options.port

    if plot_status.port is None:
        if port_name:
            message_fun('Failed to connect to iDraw2 ' + str(port_name))
        else:
            message_fun("Failed to connect to iDraw2.")
        return False
    fw_version_string = drawcore_serial.queryVersion(plot_status.port) # Full string, human readable
    fw_version_string = fw_version_string.split("DrawCore V", 1)
    fw_version_string = fw_version_string[1]
    fw_version_string = fw_version_string[:4]
    plot_status.fw_version = fw_version_string.strip() # For number comparisons
    status = drawcore_serial.query(plot_status.port, '?\r')
    if 'Alarm' in status:
        drawcore_serial.query(plot_status.port, '$X\r')
    if port_name:
        logger.debug('Connected successfully to port: ' + str(port_name))
    else:
        logger.debug(" Connected successfully")
    return True


def query_voltage(options, params, plot_status, warnings):
    """ Check that power supply is detected. """
    if params.skip_voltage_check:
        return
    return



def exhaust_queue(ad_ref):
    """
    Wait until queued motion commands have finished executing
    Uses time.sleep to sleep as long as motion commands are still executing.

    Query every 50 ms. Also break on keyboard interrupt (if configured) and
        pause button press.

    """
    return

