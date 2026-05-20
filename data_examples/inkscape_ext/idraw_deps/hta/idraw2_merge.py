"""
idraw2_merge.py
See version_string below for detailed version number.

Copyright 2023 Windell H. Oskay, All rights reserved
Evil Mad Scientist Laboratories, www.evilmadscientist.com
"""
# pylint: disable=c-extension-no-member, fixme,

from copy import deepcopy
import csv
import gettext
from io import StringIO
import os
from re import escape
from re import compile as re_compile
import time
import math
from importlib import import_module
from lxml import etree

try: # imported as a python module, e.g. imported from a wrapper (see idrawforinkscape repo)
    from drawcore_plotink.plot_utils_import import from_dependency_import
except ImportError:
    from idraw2_plot_utils_import import from_dependency_import

from idraw2_0internal import idraw    

from_dependency_import('builtins')

common_axi = from_dependency_import('idraw2_0internal.idraw_options.common_options')
common_ht = from_dependency_import('hta.hershey_options.common_options')
inkex = from_dependency_import('ink_extensions.inkex')
simplestyle = from_dependency_import('ink_extensions.simplestyle')
exit_status = from_dependency_import('ink_extensions_utils.exit_status')
ebb_serial = from_dependency_import('drawcore_plotink.drawcore_serial') 
ebb_motion = from_dependency_import('drawcore_plotink.drawcore_motion')
text_utils = from_dependency_import('drawcore_plotink.text_utils')
hershey_advanced = from_dependency_import('hta.hershey_advanced')

def parse_merge_field(field_string):
    """
    Given a string, the contents of a merge field, parse it to identify
    variable name, row offset, pre_text, post_text. (Regex does not work
    great for this particular operation.)

    field_string may be in a format as follows:
        {[[ pre_text ]]} variable_name {++##} {[[ post_text ]]}
    Where curly brackets indicate optional features.

    pre_text and post_test are strings, contained inside double
    square brackets

    White space before and after the curly braces is allowed & ignored.

    Variable name is not case sensitive, and may contain whitespace.

    pre_text, if present, must be surrounded in square brackets.
    post_text, if present, must be surrounded in square brackets.

    A row offset, if present, is labeled by "++" followed by a
    1-digit or 2-digit offset number.

    return: variable (column) name, pre_text, post_text, row_offset
    """

    stripped_input = field_string.strip() # remove leading, trailing whitespace

    pre_text = ""
    post_text = ""
    row_offset = 0

    empty_merge_field = ("", "", "", 0)

    if not stripped_input:
        return empty_merge_field

    if stripped_input[:2] == '[[':
        # The (stripped) input string starts with '[', so pre_text is likely present.
        #     Partition stripped_input at ']', splitting it at first occurrence.
        #     - If there is no following ']', then raise error to user.
        #     - If nothing is after the ']', then there is no variable name present.
        #         replace contents of merge_field with '' (empty)
        #     Otherwise:
        #         There is a valid pre_text, collect it.

        parts = stripped_input[2:].partition(']]') # drop leading '[[', split with ']]'
        if not parts[1]:    # No closing ']]' present after opening '[['
            inkex.errormsg( "Error: Improperly formatted variable field")
            inkex.errormsg( '    "{{' + field_string + '}}"')
            return empty_merge_field
        if not parts[2]:    # no variable name following the matched '}'.
            return empty_merge_field
        pre_text = parts[0]
        stripped_input = parts[2].strip()

    if stripped_input[-2:] == ']]':
        # stripped_input now contains the input string less leading/trailing whitespace,
        # and less any leading '{pre_text}' portion.
        #
        # The remaining part ends with ']]', so post_text is likely present.
        #     Reverse partition stripped_input at '[[', splitting it at first occurrence.
        #     - If there is no preceding '[[', then raise error to user.
        #     - If nothing is before the '[[', then there is no variable name present.
        #         replace contents of merge_field with '' (empty)
        #     Otherwise:
        #         There is a valid post_text, collect it.

        parts = stripped_input[:-2].rpartition('[[') # drop trailing ']]', split with '[['
        if not parts[1]:    # No opening '[[' present before closing ']]'
            inkex.errormsg( "Error: Improperly formatted variable field")
            inkex.errormsg( '    "{{' + field_string + '}}"')
            return empty_merge_field
        if not parts[0]:    # no variable name before the matched '[['.
            return empty_merge_field
        post_text = parts[2]
        stripped_input = parts[0].strip()

    if not stripped_input:
        return empty_merge_field  # If the remaining string is empty, return

    parts = stripped_input.partition('++')
    if parts[1]:
        number_tail = parts[2].strip()
        try:
            row_offset = int(number_tail)
        except ValueError:
            pass # row_offset remains at zero.

    column_name = parts[0].strip()
    if not column_name:
        return empty_merge_field

    return (column_name.lower(), pre_text, post_text, row_offset)


class iDrawMergeClass( inkex.Effect ):
    '''iDraw Merge Main Class'''

    def __init__( self, default_logging = True, params = None, ad_params = None):
        if params is None:
            params = import_module("hta.idraw2_merge_conf") # Default configuration file
        self.params = params

        if ad_params is None:
            ad_params = import_module("idraw2_0internal.idraw2_0_conf") # Default config file
        self.ad_params = ad_params # pass-through iDraw config file data

        inkex.Effect.__init__( self )

        self.OptionParser.add_option_group(
            common_axi.core_options(self.OptionParser, params.__dict__))
        self.OptionParser.add_option_group(
            common_ht.core_options(self.OptionParser, params.__dict__))
        # TODO add support for --no-rotate

        self.OptionParser.add_option("--mode", action="store", type="string", dest="mode",\
            default="single_plot",
            help="Mode or GUI tab. One of: [single_plot, auto_plot, setup, csv, text, options, "\
            + "resume]. Default: single_plot.")
        self.OptionParser.add_option( "--first_row", action="store", type="int", dest="first_row",
            default=params.first_row, help="Merge Start Row." )
        self.OptionParser.add_option( "--last_row", action="store", type="int", dest="last_row",
            default=params.last_row, help="Merge End Row." )
        self.OptionParser.add_option( "--single_type", action="store", type="string",
            dest="single_type", default="single_fix", help="The type of single-page plot to use" )
        self.OptionParser.add_option( "--single_row", action="store", type="int",
            dest="single_row", default=params.single_row, help="Specified Row to Merge" )
        self.OptionParser.add_option( "--csv_file", action="store", type="string", dest="csv_file",
            default="", help="path to the csv data file" )
        self.OptionParser.add_option( "--textmodes", action="store", type="string",
            dest="textmodes", default="none", help="Secondary GUI tab" )

        self.version_string = "iDraw2 Merge v 5.5.0, 2025-09-21"

        self.field_dict = {}
        self.merge_dict = None
        self.future_rows = [0]
        self.csv_data = None
        self.dialect_read = None
        self.row_advance = 0
        self.csv_file_path = None
        self.csv_row_count = None

        self.serial_port = None
        self.svg_data_written = False

        self.csv_file_loaded = False
        self.delay_between_rows = False # Not currently delaying between copies
        self.b_stopped = False # Not currently stopped by button press
        self.merge_error = False # No error has occurred while merging data.
        self.no_variable_fields = True
        self.require_hard_revert = False

        self.svg_rand_seed_old = int(1) #These values to be read from file
        self.svg_row_old = int(0) # Last row plotted.
        self.svg_rand_seed = int(1)
        self.svg_row = int( 1 )

        self.row_to_plot = 1
        self.max_data_row = 1
        self.last_row = 0
        self.snippet_index = 0
        self.spew_debug_data = False
        self.start_time = 0
        self.page_delays = 0
        self.rows_plotted = 0

    def effect( self ):
        '''Main entry point: check to see which mode/tab is selected, and act accordingly.'''

        self.start_time = time.time()

        skip_serial = False
        if self.options.preview:
            skip_serial = True

        # Input sanitization:
        self.options.mode = self.options.mode.strip("\"")
        self.options.single_type = self.options.single_type.strip("\"")
        self.options.font_option = self.options.font_option.strip("\"")
        if self.options.font_face is None:
            self.options.font_face = ""
        else:
            self.options.font_face = self.options.font_face.strip("\"")
        self.options.setup_type = self.options.setup_type.strip("\"")
        self.options.resume_type = self.options.resume_type.strip("\"")

        self.options.page_delay = max(self.options.page_delay, 0)

        if self.options.mode in [ "options", "text", "timing"]:
            if self.params.options_message:
                inkex.errormsg("Use the Once or Auto tab to start a new "+
                                "plot or run a preview.\n\n" +
                                "  Configuration changes are applied automatically;\n" +
                                '  Pressing "Apply" on this tab has no effect other\n' +
                                "  than displaying this message.")
            return

        if self.options.mode == "csv":
            skip_serial = True

        self.last_row = self.options.last_row
        ad = idraw.iDraw(params=self.ad_params)
        hta = hershey_advanced.HersheyAdv()

        ad.getoptions([])
        self.svg = self.document.getroot()

        ad.plot_status.resume.read_from_svg(self.svg)

        self.svg_row_old = ad.plot_status.resume.old.row # Access params from plotdata
        ad.called_externally = True

        if self.options.mode == "single_plot" and self.options.single_type == "query_row":
            # Do not plot; Query and report last row plotted
            ad.backup_original = deepcopy(self.original_document)
            self.preparse_svg() # Determine which rows are accessed.
            self.document = deepcopy(self.original_document) # Hard revert

            if self.no_variable_fields:
                note = 'No variable merge fields found in this document.\n'
                note += 'Thus, no rows of data are accessed when merging '
                note += 'and plotting this document.\n\n'
                note += 'The document reports that the last data row merged '
                note += 'was Row ' + str(int(self.svg_row_old)) + '.'
                inkex.errormsg(note)
                return
            note =  'Last data row merged: Row ' + str(int(self.svg_row_old)) + '\n'
            note += 'Next row to merge: Row ' + str(int(self.svg_row_old + 1))
            inkex.errormsg(note)

            max_future = max(self.future_rows)
            current_max = max_future + int(self.svg_row_old) + 1
            if max_future > 0:
                current_max = max_future + int(self.svg_row_old) + 1
                note = '\nThis template document accesses data from rows up through '
                note += str(max_future) + ' ahead of the current row. The next row merged '
                note += 'will access data up through Row ' + str(current_max) + '.'
                inkex.errormsg(note)

            self.read_csv(False) # Read CSV file if available, suppress error reporting.
            if not self.csv_row_count or not self.merge_dict:
                return

            if self.csv_row_count > 0:
                note = '\nThe selected CSV file has ' + str(self.csv_row_count)
                note += ' data rows.\n'
                inkex.errormsg(note)

                remaining_rows = self.csv_row_count - self.svg_row_old
                if current_max > self.csv_row_count:
                    note = 'Because of this, there is insufficient data available '
                    note += 'in the CSV file to merge additional rows.'
                    if remaining_rows > 0:
                        if remaining_rows == 1:
                            note += "\n\nOne row"
                        else:
                            note += "\n\n" + str(remaining_rows) + ' rows'
                        note += ' at the end of the CSV file will NOT be used, as there will '
                        note += 'not be enough data to fill the template document.'
                    inkex.errormsg(note)
                else:
                    rows_in_template = 1 + max_future
                    merges = math.floor (remaining_rows / rows_in_template)
                    remainder = remaining_rows % rows_in_template

                    note = 'There are enough rows remaining in the CSV file '
                    note += 'to fill this template '
                    if merges > 1:
                        note += str(merges) + ' times.'
                    else:
                        note += 'only once.'
                    if remainder > 0:
                        if remainder == 1:
                            note += "\n\nOne row"
                        else:
                            note += "\n\n" + str(remainder) + ' rows'
                        note += ' at the end of the CSV file will NOT be used, as there will '
                        note += 'not be enough data to fill the template document.'
                    inkex.errormsg(note)
            return # Exit after reporting row count data to user

        if not skip_serial:
            ad.options.port_config = self.options.port_config
            ad.options.port = self.options.port
            ad.serial_connect()
            self.serial_port = ad.plot_status.port
            if self.serial_port is None: # Note: If so, the user will be warned about the failure.
                return

        if self.options.mode != "csv":
            ad.backup_original = deepcopy(self.original_document)
            self.preparse_svg() # For cases where we will merge and plot
            self.require_hard_revert = True # True if we preparse, but haven't run iDraw to plot.

        if self.options.mode == "auto_plot":
            pen_down_inches = 0.0 # Local variable
            pen_up_inches =   0.0   # Local variable
            pt_estimate = 0.0 # Local variable
            continue_plotting = True

            self.row_to_plot = int(self.options.first_row)
            max_future = max(self.future_rows)

            if self.options.last_row == 0: # "Continue until last row of data"
                self.last_row = 100000  # A large number; only limit by size of data.
            if not self.no_variable_fields:
                self.read_csv()
                if not self.csv_row_count or not self.merge_dict or not self.csv_data:
                    continue_plotting = False

                if continue_plotting:
                    if self.row_to_plot > self.csv_row_count:
                        inkex.errormsg( gettext.gettext( \
                            "No merge data found in specified range of rows."))
                        continue_plotting = False

                    elif self.max_data_row > self.csv_row_count:
                        inkex.errormsg( "Error: Data requested beyond end of the CSV file.\n")
                        inkex.errormsg( "This merge template document specifies to use data")
                        inkex.errormsg( "from this row, number "  + str(self.row_to_plot) + \
                            ", up through row number " + str(self.max_data_row) + ".")
                        inkex.errormsg( "However, the CSV data file has only " + \
                            str(self.csv_row_count) + " rows.")
                        continue_plotting = False

                    elif self.max_data_row > self.last_row:
                        inkex.errormsg( "Error: Data requested beyond end row.\n")
                        inkex.errormsg( "This merge template document specifies to use data")
                        inkex.errormsg( "from this row, number "  + str(self.row_to_plot) + \
                            ", up through row number " + str(self.max_data_row) + ".")
                        inkex.errormsg( "However, the Merge end row is row " + \
                            str(self.options.last_row) + ".")
                        continue_plotting = False

            if self.last_row < self.options.first_row:
                if continue_plotting:
                    inkex.errormsg( 'Nothing to plot; No data rows selected.')
                    continue_plotting = False

            if continue_plotting:
                while continue_plotting:
                    self.svg_rand_seed = int(time.time()*100) # New seed for new plot
                    if self.spew_debug_data:
                        inkex.errormsg( 'Merging row number: ' + str(int(self.row_to_plot)))
                    self.merge_and_plot(hta, ad)

                    if self.merge_error:
                        continue_plotting = False
                    else:
                        pen_down_inches += ad.plot_status.stats.down_travel_inch
                        pen_up_inches += ad.plot_status.stats.up_travel_inch
                        pt_estimate += ad.plot_status.stats.pt_estimate # Local copy

                    if ad.plot_status.stopped: # Pause was triggered during the previous row.
                        inkex.errormsg( 'Paused while plotting row number ' +\
                            str(int(self.row_to_plot))  + '.')
                        continue_plotting = False
                    else:    # Finished plotting the row without being paused
                        self.row_to_plot = self.row_to_plot + 1 + max_future
                        if (self.row_to_plot + max_future) > self.last_row:
                            # Stop if we're at the end of the data.
                            # self.last_row is already truncated for future-row data usage.
                            continue_plotting = False
                        else:    # Delay before plotting next row.
                            self.preparse_svg()
                            self.read_csv()

                            time_counter = 10 * self.options.page_delay  # 100 ms units
                            self.delay_between_rows = True # Flag that we are now waiting
                            if self.spew_debug_data:
                                inkex.errormsg( 'Delaying ' + str(int(self.options.page_delay))\
                                    + ' seconds.')
                            while time_counter > 0:
                                time_counter -= 1
                                if self.b_stopped is False:
                                    self.page_delays += 100
                                    if self.options.preview:
                                        pt_estimate += 100
                                    else:
                                        time.sleep(0.100) # Short intervals for responsiveness
                                        self.pause_check() # Query if button pressed

                            self.delay_between_rows = False  # Unset delay-time flag
                            if self.b_stopped: # If button pressed
                                self.row_to_plot -= max_future + 1 # Revert "last row merged"
                                inkex.errormsg( 'Sequence halted after row number ' +\
                                    str(int(self.row_to_plot))  + '.')
                                continue_plotting = False # Cancel plotting sequence

                ad.plot_status.stats.down_travel_inch = pen_down_inches # Copy back to ad class
                ad.plot_status.stats.up_travel_inch = pen_up_inches
                ad.plot_status.stats.pt_estimate = pt_estimate
                self.print_time_report(ad)

        elif self.options.mode == "single_plot":
            do_plot_local = True
            if self.options.single_type == "single_fix": # Plot a specified row
                self.row_to_plot = int(self.options.single_row)
            elif self.options.single_type == "single_adv" : # Automatically advance
                self.row_to_plot = int(self.svg_row_old + 1)
            else:
                do_plot_local = False

            self.svg_rand_seed = int(time.time()*100) # New random seed for new plot
            self.last_row = self.row_to_plot # Last row is equal to first row, in this case.

            if not self.no_variable_fields:
                self.read_csv()
                if not self.csv_row_count or not self.merge_dict or not self.csv_data:
                    do_plot_local = False

                if do_plot_local:
                    if self.row_to_plot > self.csv_row_count:
                        do_plot_local = False
                        inkex.errormsg( gettext.gettext( \
                            "No merge data found in row number " ) +\
                                str(self.row_to_plot) + '.')
                    elif self.max_data_row > self.csv_row_count:
                        inkex.errormsg( "Error: Data requested beyond end of the CSV file.\n")
                        inkex.errormsg( "This merge template document specifies to use data")
                        inkex.errormsg( "from this row (row "  + str(self.row_to_plot) + \
                            ") up through row number " + str(self.max_data_row) + ".")
                        inkex.errormsg( "However, the CSV data file has only " + \
                            str(self.csv_row_count) + " rows.")
                        do_plot_local = False
            if do_plot_local:
                self.merge_and_plot(hta, ad)
                self.print_time_report(ad)

        elif self.options.mode == "resume":
            ad.options.mode = "resume"
            self.svg_rand_seed = ad.plot_status.resume.old.rand_seed # Preserve random seed
            self.row_to_plot = self.svg_row_old         # Preserve SVG Row
            ad.options.resume_type = self.options.resume_type

            if self.options.resume_type == "home":
                self.options.font_option = "none"    # Disable Hershey Advanced substitution
                self.max_data_row = self.row_to_plot # Ensure that row is saved to SVG file.
                self.merge_and_plot(hta, ad)
            elif ad.plot_status.resume.old.application != "idraw2 merge":
                inkex.errormsg( gettext.gettext( "No iDraw2 Merge resume data found in file." ))
            elif ad.plot_status.resume.old.layer < 0: # A paused "all layers" plot
                self.last_row = self.row_to_plot
                self.read_csv()

                if (self.csv_row_count is not None) or self.no_variable_fields:
                    self.merge_and_plot(hta, ad)
                    self.print_time_report(ad)
            else:
                inkex.errormsg( gettext.gettext( "No in-progress plot data found saved in file." ))

        elif self.options.mode == "setup":
            if self.options.preview:
                inkex.errormsg( gettext.gettext('Command unavailable while in preview mode.'))
            else:
                ad.options.mode = "setup"
                ad.options.setup_type = self.options.setup_type
                selected_options = {item: self.options.__dict__[item] for item in ['pen_pos_up',
                    'pen_pos_down', 'pen_rate_raise', 'pen_rate_lower',]}
                ad.options.__dict__.update(selected_options)
                ad.document = self.document
                ad.options.port = self.serial_port
                ad.options.port_config = 2
                ad.effect()

        elif self.options.mode == "csv":
            self.read_csv()

        ad.warnings.report(False, inkex.errormsg) # print warnings, e.g., low voltage

        if self.serial_port is not None:
            ebb_motion.doTimedPause(self.serial_port, 10, False) # Pause for commands to finish.
            ebb_serial.closePort(self.serial_port)

        if self.require_hard_revert:
            self.document = deepcopy(self.original_document) # Hard revert

    def recursive_read_text(self, node_list):
        '''Recursively collect contents of a text node'''
        for node in node_list:
            if node.text is not None:
                self.text_str += node.text
                self.original_snippets.append(node.text)
            if node.tag in [inkex.addNS('tspan', 'svg'), 'tspan',
                inkex.addNS('flowPara', 'svg'), 'flowPara',
                inkex.addNS('flowSpan', 'svg'), 'flowSpan',]:
                self.recursive_read_text(node)
            if node.tail is not None:
                self.text_str += node.tail
                self.original_snippets.append(node.tail)

    def recursive_replace_text(self, node_list, replacement_list):
        """
        Perform in-situ text replacement of merge fields, replacing merge fields with standardized
        forms in a locally-flat part of the SVG tree, such that the merge field text appears as
        plain text in the SVG document. The replacement_list input specifies the start and end
        indexes where the original text is located, as well as the new text that replaces it.
        """
        for node in node_list:
            if node.text is not None:
                node.text = self.new_snippets[self.snippet_index]
                self.snippet_index += 1
            if node.tag in [inkex.addNS('tspan', 'svg'), 'tspan',
                inkex.addNS('flowPara', 'svg'), 'flowPara',
                inkex.addNS('flowSpan', 'svg'), 'flowSpan',]:
                self.recursive_replace_text(node, replacement_list)
            if node.tail is not None:
                node.tail = self.new_snippets[self.snippet_index]
                self.snippet_index += 1

    def preparse_recursive_traverse( self , node_list):
        """ Recursively traverse the document, looking for merge fields """
        for node in node_list:
            # Ignore locked and invisible nodes, as well as documentation layers
            locked_var = node.get(inkex.addNS('insensitive', 'sodipodi'))
            if locked_var == "true":
                continue # Object is locked; skip

            node_style_parsed = simplestyle.parseStyle(node.get('style'))

            if 'display' in node_style_parsed.keys() and node_style_parsed['display'] == 'none':
                continue  # Do not parse this object or its children

            if node.get('display') == 'none': # Node may have a display="none" attribute as well
                continue  # Do not parse this object or its children

            if node.tag == inkex.addNS('g', 'svg') or node.tag == 'g':
                ink_label = node.get(inkex.addNS('label', 'inkscape'))
                if not ink_label:
                    pass
                else:
                    if ink_label == 'hta':
                        continue    # Do not traverse groups of rendered text.
                if node.get(inkex.addNS('groupmode', 'inkscape')) == 'layer':
                    # This group is an Inkscape layer
                    layer_name = node.get(inkex.addNS('label', 'inkscape'))
                    if layer_name:
                        if layer_name[0] == '%': # Documentation layer
                            continue    # Do not traverse documentation layer.
                self.preparse_recursive_traverse(node)

            elif (node.tag == inkex.addNS('text', 'svg')) or (node.tag == 'text') or\
                                        (node.tag == inkex.addNS("flowRoot", "svg")):
                self.text_str = ""
                self.original_snippets = []
                # First pass through text node:
                # Compile "plain text" string of the contents of this text node:
                self.recursive_read_text(node)

                # Check the plain string to see if it contains any merge fields.
                # If so, note their start and end positions within the string and parse them.
                field_replace_dict = []
                text_pos = 0
                while True: # Until we finish checking for merge fields
                    index_start = self.text_str.find('{{', text_pos) # Look for merge field starts
                    if index_start < 0:
                        break # No start of merge field found; exit loop

                    # Look for end of merge field, starting 3 positions later: two for '{{' plus
                    # a minimum of one character between the opening and closing brackets.
                    text_pos = index_start + 3
                    index_end = self.text_str.find('}}', text_pos)
                    if index_end < 0:
                        break # No end of merge field found; exit loop
                    index_end += 2 # Advance 2 for the closing '}}'

                    text_pos = index_end # Prepare for looking for next merge field

                    # A merge field "{{example}}" as it currently appears in the plain text:
                    merge_field_old = self.text_str[index_start:index_end]
                    field_name_old = merge_field_old[2:-2] # contents inside curly braces

                    # Case-insensitive match for {{nextrow}}:
                    if field_name_old.lower() == "nextrow":
                        self.row_advance += 1 # Advance to next row
                        # We will replace {{nextrow}} with empty string in the text:
                        field_replace_dict.append((index_start, index_end, ""))
                        continue # Find next merge field in this text block

                    field_elements = parse_merge_field(field_name_old)
                    # parse_merge_field output: column_name, pre_text, post_text, row_offset

                    # Rewrite the merge field in a standardized format, with whitespace stripped,
                    # lowercase column name, and additional flags ([[x]], ++) only where necessary.
                    # And, populate a list of the flags now, to avoid to re-parsing later.

                    if field_elements[0]: # If column name is not empty
                        self.no_variable_fields = False
                        merge_field_new = '{{'
                        if field_elements[1]:
                            merge_field_new += '[[' + field_elements[1] + ']]'
                        merge_field_new += field_elements[0]
                        offset = self.row_advance + field_elements[3]
                        if offset > 0:
                            merge_field_new += '++' + str(offset)
                        if field_elements[2]:
                            merge_field_new += '[[' + field_elements[2] + ']]'
                        merge_field_new += '}}'

                        # Add field to substitution dict, replacing any existing entry:
                        self.field_dict[merge_field_new] = (field_elements[0], \
                                field_elements[1], field_elements[2], offset)

                        if offset not in self.future_rows:
                            self.future_rows.append(offset)

                        # Replace the original field text with the new text:
                        field_replace_dict.append((index_start, index_end, merge_field_new))

                # Replace merge fields in document with properly formatted replacements,
                # and flatten them so that they appear in the SVG when converted to a string.
                if len(field_replace_dict) > 0:
                    self.new_snippets = []
                    # Step through text snippets, making the replacements in order
                    strpos = 0
                    stage = 0
                    start = field_replace_dict[stage][0]
                    stop = field_replace_dict[stage][1]
                    for block in self.original_snippets:
                        new_snip = ""
                        for char in block:
                            if strpos < start:
                                new_snip += char
                            elif strpos == start:
                                new_snip += field_replace_dict[stage][2]
                            elif strpos >= (stop - 1):
                                stage += 1
                                if stage >= len(field_replace_dict):
                                    start = len(self.text_str) + 1
                                    stop = start + 1
                                else:
                                    start = field_replace_dict[stage][0]
                                    stop = field_replace_dict[stage][1]
                            strpos += 1
                        self.new_snippets.append(new_snip)
                    del self.original_snippets # Remove old list
                    self.snippet_index = 0
                    self.recursive_replace_text(node, field_replace_dict)


    def preparse_svg( self ):
        """
        Pre-parse SVG document to build a dictionary of the merge fields present.

        Lays groundwork for populating dictionaries for variable substitution.

        Parse merge fields, rewrite them in a standard format, and move them
        to fit within a single text element so that they exist in place when
        the SVG tree is flattened into a string.
        """

        self.preparse_recursive_traverse( self.document.getroot() )
        self.future_rows.sort()
        # inkex.errormsg( "self.field_dict:\n" + str(self.field_dict)) # Optional debug
        # inkex.errormsg( "self.future_rows:\n" + str(self.future_rows)) # Optional debug


    def read_csv( self, verbose=True ):
        '''
        *   When loading a new CSV file or checking the selected file:
            - Load CSV file
            - Determine format
            - Count the rows
            - Report results to user

        *   Before merging first row of data:
            - Load CSV file
            - Determine format
            - Count the rows
            - Limit range of automatic advance

        *   For any row (including the first):
            - Initialize a dictionary with column header names as keys and values
                for the row(s) to be merged.
        '''

        if self.no_variable_fields and self.options.mode != "csv":
            # Do not load CSV file if there are no variable fields in the SVG template
            return
        if not self.csv_file_loaded: # Only read the file the first time that read_csv()  is called
            self.csv_data = None
            file_path = self.options.csv_file
            if not file_path:
                if verbose:
                    inkex.errormsg( "No CSV file selected. Use Data tab to select a CSV file." )
                return
            self.csv_file_path = file_path   # Path & Name of the file

            try:
                with open(file_path, mode='r', encoding='utf-8-sig') as file_ref:
                    self.csv_data = file_ref.read()
            except UnicodeDecodeError:
                if verbose:
                    inkex.errormsg("Unable to determine encoding of CSV data file." \
                    + "\nBe sure to save the data file as a (UTF-8) CSV file.")
            except:
                if verbose:
                    inkex.errormsg("Unable to load CSV data file.\nUse Data tab to select file.")

            if not self.csv_data:
                return
            self.csv_file_loaded = True
            self.csv_data  = '\n'.join(self.csv_data.splitlines())
            self.dialect_read = csv.Sniffer().sniff(StringIO(self.csv_data).readline())
            self.dialect_read.doublequote = True
            # This option forces two quotes ("") to be read as an escaped quotation mark; a hack
            #   for excel compatibility; it may cause issues with some less common encodings.

            reader_ref = csv.reader(StringIO(self.csv_data), self.dialect_read)
            self.csv_row_count = sum(1 for row in reader_ref) - 1 # Subtract 1 for header row

            # This count exhausts the reader by iteration; we need to reset the reader:
            reader_ref = csv.DictReader(StringIO(self.csv_data), dialect=self.dialect_read)
            csv_column_count = len(reader_ref.fieldnames)

            if verbose and self.options.mode == "csv": # Report CSV structure only
                if (self.csv_row_count > 0) and (csv_column_count > 0):
                    note = "CSV data file contains " + str(self.csv_row_count)
                    if self.csv_row_count == 1:
                        note += " row of data.\r\r"
                    else:
                        note += " rows of data.\r\r"
                    inkex.errormsg(note)

                    note = "CSV data file contains " + str(csv_column_count)
                    if csv_column_count == 1:
                        note += " column of data:\r"
                    else:
                        note += " columns of data:\r"
                    inkex.errormsg(note)

                    key_names = ""
                    for item in reader_ref.fieldnames:
                        key_names += item + ", "
                    key_names = key_names[:-2]  # drop last two characters from string (", ")
                    inkex.errormsg( key_names ) # Print key list for user
                else: # self.csv_row_count is not > 0.
                    inkex.errormsg( "Unable to interpret selected file" +\
                        str(os.path.basename(file_path)) + ".")
                return # Remaining parts of read_csv() are not relevant to "csv" mode.

            practical_row_count = self.csv_row_count - max(self.future_rows)
            if practical_row_count < self.last_row:
                self.last_row = practical_row_count # Limit last row of data to end of CSV file

            # Check that every merge field corresponds to a column name in the CSV file
            field_names = ["nextrow"]
            for name in reader_ref.fieldnames:
                field_names.append(name.lower().strip())
            for field_values in self.field_dict.values():
                column_name = field_values[0].lower().strip()
                if column_name not in field_names:
                    inkex.errormsg( 'Data error: The template file contains a merge field ("' +\
                        str(column_name) + '") that does not appear in the CSV file.')
                    return
        # This count exhausts the reader by iteration; Reset the reader:
        reader_ref = csv.DictReader(StringIO(self.csv_data), dialect=self.dialect_read)

        # Determine maximum row that we need to have data for in merging _this row_.
        self.max_data_row = self.row_to_plot + max(self.future_rows)
        if self.max_data_row > self.csv_row_count:
            return # If need data past the end of the CSV file, return

        self.merge_dict = {} # Initialize the dictionary of substitutions. Dict includes data for
            # self.row_to_plot, as well as from "future" rows adddressed via the "++" syntax.

        current_row = 1
        row = next(reader_ref)

        for data_row_offset in self.future_rows:
            row_local = self.row_to_plot + data_row_offset

            while current_row < row_local: # Advance to next row that we need data from
                row = next(reader_ref)
                current_row += 1

            row_dict = {} # Dictionary of row elements, lowercase
            for field in reader_ref.fieldnames:
                row_dict[field.lower().strip()] = row[field]

            for field_string in self.field_dict:
                field_elements = self.field_dict[field_string]
                # field_elements -> (column_name, pre_text, post_text, row_offset)

                if data_row_offset == field_elements[3]:    # If row is matched
                    if field_elements[0] in row_dict:       # If (lowercase) field name matches
                        replacement_text = ""
                        row_text = row_dict[field_elements[0]]

                        if row_text:    # If row_text is not empty
                            # Replace returns _within_ returned text by '\\r':
                            row_text =  row_text.replace('\r\n','\\r') # Win
                            row_text =  row_text.replace('\n','\\r')  # Mac/Linux
                            replacement_text = field_elements[1] + row_text + field_elements[2]
                        else:
                            replacement_text = ''

                        self.merge_dict[text_utils.xml_escape(field_string)] =\
                                        text_utils.xml_escape(replacement_text)
        # inkex.errormsg("self.merge_dict: " + str(self.merge_dict))# Debug: print merge dictionary


    def data_merge(self):
        ''' Perform regular expression data substitution'''
        verbose = False
        if not self.merge_dict:
            return
        regex = re_compile("(%s)" % "|".join(map(escape, self.merge_dict.keys())))
        xmlstr = etree.tostring(self.document, encoding='unicode', method='xml')

        try:
            result = regex.sub(lambda mo: self.merge_dict[mo.string[mo.start():mo.end()]], xmlstr)
            del xmlstr # delete this string, which may be large.
            parser_ref = etree.XMLParser(huge_tree=True)
            self.document = etree.parse(StringIO(result), parser=parser_ref)
        except KeyError:
            if not self.merge_error:
                message = 'Error performing merge operation; plot aborted.'
                message += '\nCheck to make sure that your variable fields match.'
                message += 'those in your CSV data file. In case of '
                message += 'difficultly, please contact technical support.'
                inkex.errormsg( message )
            if verbose:
                inkex.errormsg( '\nMerge list:' + str(self.merge_dict))
            self.merge_error = True
        except:
            if not self.merge_error:
                inkex.errormsg("\nAn error occurred while merging data; plot aborted.")
            self.merge_error = True


    def merge_and_plot( self, ha_ref, ad_ref ):
        '''
        Merge and plot the actual SVG document, if so selected in the interface.

        Order of operations:
        0. Make a backup copy of the document
        1. Merge the correct row of text data, if selected
        2. Perform SVG font substitution, if selected, using Hershey Advanced
            * If plotting a new document, use new random seed
            * If resuming, use random seed from document.
        3. Plot the document, using idraw.py
            * We do not actually need to send the random seed to idraw.py.
            * idraw.py will replace SVG data with _original_ (backup) SVG data
              before appending preview and progress data.
        4. Collect plotdata output from idraw.py and modify
            * Save merge row number (perhaps in place of layer)
            * Save random seed that we used for Hershey Advanced, overruling that in the plotdata.

        For each row to be merged:
            * Update dictionary
            * Use regular expressions _once_ to replace text via dictionary
                *    Start with string version of etree
                *    Convert back to etree
        '''

        if ((self.options.mode == "resume") and ( self.options.resume_type == "home" )):
            pass    # perform no merging
        elif self.no_variable_fields:
            pass    # perform no merging
        else:
            self.data_merge() # Execute the merge operation
        if self.merge_error:
            return

        if self.options.font_option != "none":
            ha_ref.getoptions([])

            selected_options = {item: self.options.__dict__[item] for item in ['font_option',
                'font_face', 'enable_defects', 'leading_var', 'baseline_var', 'indent_var',
                'kern_var', 'size_var', 'letter_spacing', 'word_spacing',]}
            ha_ref.options.__dict__.update(selected_options)
            ha_ref.options.rand_seed        = self.svg_rand_seed
            ha_ref.document                 = self.document
            ha_ref.options.preserve_text    = False
            ha_ref.effect() # Perform Hershey Advanced substitution

        # 3. Plot the document, using idraw.py. Many options to pass through.
        # Note that ad_ref.options.mode defaults to "plot" ; we do not actively set it.
        selected_options = {item: self.options.__dict__[item] for item in ['pen_pos_up',
            'pen_pos_down', 'speed_pendown', 'speed_penup', 'accel', 'pen_rate_raise',
            'pen_rate_lower', 'pen_delay_up', 'pen_delay_down', 'auto_rotate', 'const_speed',
            'report_time', 'resolution', 'preview', 'rendering', 'model',
            'reordering', 'random_start', 'hiding', 'webhook', 'webhook_url',]}
        ad_ref.options.__dict__.update(selected_options)
        ad_ref.options.copies = 1
        ad_ref.params = self.ad_params

        if self.options.font_option != "none":
            ad_ref.warnings.suppress('text') # suppress "unconverted text" warnings

        if not self.options.preview:
            ad_ref.options.port = self.serial_port
            ad_ref.options.port_config = 2
        ad_ref.document = self.document # Pass the document off for plotting

        ad_ref.effect() # Plot the document using idraw.py
        self.rows_plotted += 1
        self.require_hard_revert = False

        # 4. Collect plotdata output from idraw.py and modify
        # Retrieve the modified version of the document, which may contain
        # updated data, such as the preview and/or save data
        self.document = ad_ref.document

        self.svg_data_written = False
        self.modify_plotdata() # Save merge row number and random seed

    def pause_check (self):
        ''' Check to see if plot stopped while between copies.'''
        if self.b_stopped:
            return # We have _already_ halted the plot due to a button press. No need to proceed.
        if self.options.preview:
            str_button = ['0']
        else:
            str_button = ebb_motion.QueryPRGButton(self.serial_port, False) #Query button
        try:
            pause_state = str_button[0]
        except:
            inkex.errormsg( '\nUSB Connectivity lost.')
            pause_state = '2' # Pause the plot; we appear to have lost connectivity.
            if self.spew_debug_data:
                inkex.errormsg( '\n USB Connectivity lost' )

        if ((pause_state == '1') and (self.delay_between_rows is False)):
            if self.spew_debug_data:
                inkex.errormsg( '\n Paused by button press. ' )

        if pause_state in ('1', '2'):  # Stop plot
            if self.delay_between_rows is False: # No message if paused while waiting between rows.
                inkex.errormsg( 'Use the "resume" feature to continue.' )
            self.b_stopped = True


    # ad.plot_status.resume.old.row
    def modify_plotdata( self ):
        ''' Update parameters saved in SVG file that are specific to merging. '''
        data_node = None
        nodes = self.document.xpath('plotdata')
        if nodes:
            data_node = nodes[0]
        if data_node is not None:
            try:
                data_node.set('application', "idraw2 merge")      # Name of this program
                data_node.set('row', str(self.max_data_row))       # Data merge row number
                data_node.set('rand_seed', str(self.svg_rand_seed)) # Random seed for this row
                self.svg_data_written = True
            except TypeError:
                pass  # Leave as default if not found

    def print_time_report( self, ad_ref ):
        '''Report elapsed time to user'''
        if self.rows_plotted == 0:
            return
        if not self.options.report_time:
            return

        if self.rows_plotted > 1:
            if self.options.preview:
                inkex.errormsg("Total rows to merge and plot: %d." % self.rows_plotted)
            else:
                inkex.errormsg("Total rows plotted: %d." % self.rows_plotted)

        elapsed_time = time.time() - self.start_time
        ad_ref.plot_status.stats.page_delays = self.page_delays
        ad_ref.plot_status.stats.report(ad_ref.options, inkex.errormsg, elapsed_time)
        ad_ref.pen.status.report(ad_ref, inkex.errormsg)

if __name__ == '__main__':
    e = iDrawMergeClass()
    exit_status.run(e.affect)
