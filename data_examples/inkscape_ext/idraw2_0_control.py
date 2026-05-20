'''
Automatically generated dumb wrapper to call idraw2_0internal.idraw_control.idraw_control as an inkscape extension.
To regenerate, run python bin/generatewrappers.py
'''
import logging

from lxml import etree

from idraw_plot_utils_import import from_dependency_import
idraw_control = from_dependency_import('idraw2_0internal.idraw_control')
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
        conf = import_module("idraw2_0_conf")
        e = idraw_control.iDrawWrapperClass(params=conf, default_logging=False)
    except ImportError as ie:
        if "idraw_conf" == "notamodule":
            # assuming everything is going well, this just means there is no config or logging assigned in the generatewrappers.py script
            e = idraw_control.iDrawWrapperClass()
        else:
            raise
    exit_status.run(e.affect)
