#!/usr/bin/env python
# coding=utf-8
'''
Hershey Advanced Python API wrapper

Copyright 2023 Windell H. Oskay, All rights reserved
Evil Mad Scientist Laboratories, www.evilmadscientist.com

'''

import sys
from lxml import etree

from .hershey_advanced import HersheyAdv

from .plot_utils_import import from_dependency_import
inkex = from_dependency_import('ink_extensions.inkex')
plot_utils = from_dependency_import('drawcore_plotink.plot_utils') 

class Setup(HersheyAdv):
    """Main class for Hershey Advanced python API"""

    def __init__(self, svg_input=None):
        """Initialize class, load options, and call __init__ from HersheyAdv"""
        HersheyAdv.__init__(self)

        inkex.localize()
        self.getoptions([])
        self.load_svg_doc(svg_input)


    def load_svg_doc(self, svg_input=None):
        """Parse template SVG file"""
        file_ok = False

        if svg_input is None: # If no other input provided, use trivial SVG
            svg_input = plot_utils.trivial_svg # Minimal blank A4 document

        try: # Parse input file or SVG string
            stream = open(svg_input, encoding="utf8")
            pars = etree.XMLParser(huge_tree=True)
            self.document = etree.parse(stream, parser=pars)
            stream.close()
            file_ok = True
        except IOError:
            pass # It wasn't a file...
        if not file_ok:
            try:
                svg_string = svg_input.encode('utf-8') # Need consistent encoding.
                pars = etree.XMLParser(huge_tree=True, encoding='utf-8')
                self.document = etree.ElementTree(etree.fromstring(svg_string, parser=pars))
                file_ok = True
            except:
                inkex.errormsg("Unable to open SVG input file.")
                # sys.exit()
        if not file_ok:
            raise RuntimeError("Unable to open SVG input file")


    def list_fonts(self):
        """ Return a list of available SVG fonts. """
        mode_temp = self.options.mode # Stash user's mode selection temporarily
        self.options.mode = "list_fonts"
        self.effect()
        self.options.mode = mode_temp
        return self.font_list


    def run(self):
        """ Execute the required action and return the resulting SVG. """
        if self.document is None:
            inkex.errormsg("No SVG input provided.")
            inkex.errormsg("Use Setup(svg_input) or load_svg_doc(svg_input) before run.")
            raise RuntimeError("No SVG input provided.")
            return None
        self.effect()

        if self.error_code == 100:
            raise RuntimeError("Scriptalizer failure; Unable to reach server.")
        if self.error_code == 101:
            raise RuntimeError("Scriptalizer failure; No response received.")
        if self.error_code == 102:
            raise RuntimeError("Scriptalizer failure; bad response.")
        if self.error_code == 103:
            raise RuntimeError("Scriptalizer returned an error")

        """Return serialized copy of svg document output"""
        result = etree.tostring(self.document)
        return result.decode("utf-8")
