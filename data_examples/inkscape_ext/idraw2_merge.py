'''
Dumb wrapper to call hta.axidraw_merge as an inkscape extension.

Forked from the axidraw_control and hershey_advanced wrappers. Initial version.
'''
import logging

from lxml import etree

from idraw_plot_utils_import import from_dependency_import
idraw_merge = from_dependency_import('hta.idraw2_merge')
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
        conf = import_module("idraw2_0_conf") # Import axidraw_conf.py from top level of extensions
        merge_conf = import_module("idraw2_merge_conf") # Import axidraw_conf.py from top level of extensions
        e = idraw_merge.iDrawMergeClass(ad_params=conf, params=merge_conf, default_logging=False)
    except ImportError as ie:
        if "axidraw_conf" == "notamodule":
            # assuming everything is going well, this just means there is no config or logging assigned in the generatewrappers.py script
            e = idraw_merge.iDrawMergeClass(default_logging=False)
        else:
            raise
    exit_status.run(e.affect)
