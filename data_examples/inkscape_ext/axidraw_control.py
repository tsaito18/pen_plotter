'''
Automatically generated launcher to launch axidraw_control.
This launcher is used for scripts that are built via pyinstaller (e.g. axidraw_control.py, axidraw_naming.py).
To regenerate, run python bin/generatewrappers.py
'''
'''
import subprocess
import sys

command = ['./build_deps/axidraw_control'] + sys.argv[1:]
proc = subprocess.run(command, capture_output=True, text=True)

# print error messages, if there are any
if proc.stderr != "":
    sys.stderr.write(proc.stderr)

# inkscape parses stdout for the result of an extension
sys.stdout.write(proc.stdout)
'''

'''
Automatically generated dumb wrapper to call axidrawinternal.axidraw_control.axidraw_control as an inkscape extension.
To regenerate, run python bin/generatewrappers.py
'''
import logging

from lxml import etree

from plot_utils_import import from_dependency_import
axidraw_control = from_dependency_import('axidrawinternal.axidraw_control')
exit_status = from_dependency_import('ink_extensions_utils.exit_status')
message = from_dependency_import('ink_extensions_utils.message')

root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)
root_logger.addHandler(message.UserMessageHandler()) # to stderr/inkscape "has received additional data" window
# consider adding a handler to send logs to extension-errors.log?

if __name__ == '__main__':
    conf = None
    e = None # effect
    try:
        from importlib import import_module
        conf = import_module("axidraw_conf")
        e = axidraw_control.AxiDrawWrapperClass(params=conf, default_logging=False)
    except ImportError as ie:
        if "axidraw_conf" == "notamodule":
            # assuming everything is going well, this just means there is no config or logging assigned in the generatewrappers.py script
            e = axidraw_control.AxiDrawWrapperClass()
        else:
            raise
    exit_status.run(e.affect)
    # input_file = r'D:\drawing.svg'
    # exit_status.run(e.affect,[input_file])
    
