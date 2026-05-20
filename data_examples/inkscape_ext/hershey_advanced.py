'''
Dumb wrapper to call hta.hershey_advanced as an inkscape extension.
Created with the help of axidrawforinkscape's generate_extensions.py script.
This goes directly in inkscape's extensions folder along with hershey_advanced.inx and the contents of axidraw_for_inkscape*_release.
Probably would be good to make an automated pipeline to do all that.
'''
import logging
from types import SimpleNamespace

from lxml import etree

from plot_utils_import import from_dependency_import
hershey_advanced = from_dependency_import('hta.hershey_advanced')
exit_status = from_dependency_import('ink_extensions_utils.exit_status')
message = from_dependency_import('ink_extensions_utils.message')

root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)
root_logger.addHandler(message.UserMessageHandler()) # to stderr/inkscape "has received additional data" window
# consider adding a handler to send logs to extension-errors.log?

if __name__ == '__main__':
    conf = None
    e = None # effect
    conf_name = 'hershey_conf.py'
    try:
        conf_dict = dict()
        exec(open(conf_name).read(), dict(), conf_dict)
        conf = SimpleNamespace(**conf_dict)
        e = hershey_advanced.HersheyAdv(params=conf)
    except (FileNotFoundError, ImportError) as err:
        root_logger.warning(
                "{} not found in base dir, using axidraw_deps/hta/hershey_advanced_conf.py".
                format(conf_name))
        e = hershey_advanced.HersheyAdv()

    exit_status.run(e.affect)
