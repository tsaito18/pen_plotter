'''
Hershey Advanced

See version_string below for current version and date.

Copyright 2023 Windell H. Oskay, All rights reserved
Evil Mad Scientist Laboratories, www.evilmadscientist.com

Hershey Advanced uses SVG fonts, rather than the historical Hershey format.
- SVG fonts support unicode, meaning that they can use a full range of
    characters.
- Arbitrary curves are supported within glyphs; we are no longer limited to
    the straight line segments used in the historical Hershey format.

Some notes about SVG fonts:
- While SVG fonts are not typically used on their own, SVG fonts are
    an allowed "table" within an OpenType font. OTF fonts with SVG content
    are called OTF+SVG, or sometimes "Color" fonts. SVG fonts may be
    comprised of fills (regular outline fonts) or be stroke-based fonts.
- Programs like FontTools ( https://github.com/fonttools/fonttools ) can
    be used to manipulate and examine the contents of OTF fonts.
- Programs like FontForge ( http://fontforge.github.io ) can be used to
    create SVG fonts, or convert from TrueType to SVG format.
- Note that most graphical applications do not at present support
    OTF+SVG fonts. Some that do include Adobe Illustrator and Pages
    on Macs. Applications that do not support OTF+SVG will instead display
    a TrueType "fallback" typeface bundled within the OTF font bundle.
- Applications that do support editing SVG fonts may _or may not_ support
    stroke fonts within that context. Some do, some do not. Those that
    otherwise support
'''

from copy import deepcopy
from importlib import import_module
import math
import os
import random
import sys
import time

from lxml import etree

from .hershey_options import common_options

from .plot_utils_import import from_dependency_import
inkex = from_dependency_import('ink_extensions.inkex')
simplestyle = from_dependency_import('ink_extensions.simplestyle')
simpletransform = from_dependency_import('ink_extensions.simpletransform')
exit_status = from_dependency_import('ink_extensions_utils.exit_status')
plot_utils = from_dependency_import('drawcore_plotink.plot_utils') 
hta_utils = from_dependency_import('hta.hta_utils')

# DEBUG = True
DEBUG = False

class HersheyAdv(inkex.Effect):
    """ Hershey Advanced main class """
    def __init__(self, params=None):
        if params is None:
            # use default configuration file
            params = import_module("hta.hershey_conf")
        self.params = params

        inkex.Effect.__init__(self)

        self.OptionParser.add_option_group(
            common_options.core_options(self.OptionParser, params.__dict__))
        self.OptionParser.add_option_group(
            common_options.extra_options(self.OptionParser, params.__dict__))

        self.OptionParser.add_option("--useGUI", \
            action="store", type="inkbool", dest="inkscape", \
            default=False, \
            help="True if called from within Inkscape")

        self.OptionParser.add_option("--tab", \
            action="store", type="string", dest="mode", \
            default="render", \
            help="The active tab or mode when Apply was pressed")

        self.version_string = "5.5.1" # Dated 2023-05-11
        self.text_delimiter = '🐈🐈🐈🐈🐈'
        self.text_empty = ''
        self.font_list = []
        self.line_offset = 0
        self.error_code = 0

        # When using Scriptalizer, revert text back to pre-scriptalized after rendering
        #   when preserve original text option is also true. This is in the class init
        #   to facilitate overrides by functions calling effect() externally.
        self.revert_script_text = True


    def id_search(self, the_node_list):
        """Ensure that all nodes have an ID"""
        for node in the_node_list:
            the_id = node.get('id')
            if not the_id:
                if node.tag in [etree.Comment]:
                    continue
                the_id = self.id_new()
                node.set('id', the_id)
            self.doc_id_list[the_id] = 1
            if node.tag == inkex.addNS('g', 'svg') or node.tag == 'g':
                self.id_search(node)


    def id_new(self):
        """Generate a new and unique id"""
        the_id = 'a'
        while the_id in self.doc_id_list:
            the_id += random.choice('TheFiveBoxingWizardsJumpQuickly')
        self.doc_id_list[the_id] = 1
        return the_id


    def load_font(self, fontname):
        '''
        Attempt to load an SVG font from a file in our list of (likely) SVG font files.
        If we can, add the contents to the font library. Otherwise, add a "None" entry.
        '''

        if DEBUG:
            inkex.errormsg('load_font: ' )

        if fontname is None:
            return
        if fontname in self.font_dict: # The font is already loaded.
            return

        if fontname in self.font_file_list:
            the_path = self.font_file_list[fontname]
        else:
            if DEBUG:
                inkex.errormsg('[load_font]: Font ' + fontname + ' not found in font_file_list')
            self.font_dict[fontname] = None
            return # Font not located.
        try:
            if DEBUG:
                inkex.errormsg('[load_font] Trying path: ' + str(the_path))

            '''
            Check to see if there is an SVG font file for us to read.

            At present, only one font file will be read per font family; the name of the file
            must be FONT_NAME.svg, where FONT_NAME is the name of the font family.

            Only the first font found in the font file will be read. Multiple weights and
            styles within a font family are not presently supported.
            '''
            file_ref = open(the_path, encoding="utf8")
            parse_ref = etree.XMLParser(huge_tree=True)
            font_svg = etree.parse(file_ref, parser=parse_ref)
            file_ref.close()
            self.font_dict[fontname] = hta_utils.parse_svg_font(font_svg.getroot())

        except IOError:
            if DEBUG:
                inkex.errormsg('Unable to read file: ' + str(the_path))
            self.font_dict[fontname] = None
        except etree.XMLSyntaxError:
            inkex.errormsg('An error occurred while parsing SVG font at ' + str(the_path))
            self.font_dict[fontname] = None
        except:
            inkex.errormsg('Unable to load and parse SVG font at ' + str(the_path))
            self.font_dict[fontname] = None

    def font_table(self):
        """Generate display table of all available SVG fonts"""

        for fontname in self.font_file_list:
            self.load_font(fontname) # Todo, for future: Load only one font at a time.

        if self.options.mode == "list_fonts": # Populate font_list for the Python API
            self.font_list = sorted(self.font_dict.keys())
            return

        font_size_temp = 0.13 # in inches -- will be scaled by viewbox factor.
        font_size_text = str(font_size_temp / self.vb_scale) + 'px'
        labeltext_style = simplestyle.formatStyle({'stroke' : 'none',\
         'font-size':font_size_text, 'fill' : 'black', \
                'font-family' : 'sans-serif', 'text-anchor': 'end'})

        x_centering = 0.3 * self.svg_width / self.vb_scale # center ~1/3 from left on page
        x_offset = font_size_temp / self.vb_scale
        y_offset = 1.5 * x_offset
        y_current = y_offset
        x_offset += x_centering # Begin offset from center position

        txt_grp = inkex.etree.SubElement(self.current_layer, 'g') # Embed text in a group
        txt_grp.set('id', self.id_new())

        for fontname in sorted(self.font_dict):
            text_attribs = {'x': str(x_centering), 'y': str(y_current), 'hta-ignore':'true'}
            textline = inkex.etree.SubElement(txt_grp, inkex.addNS('text', 'svg'), text_attribs)
            textline.text = fontname
            textline.set('style', labeltext_style)
            text_attribs = {'x':str(x_offset), 'y': str(y_current)}
            sampletext_style = {'stroke':'none', 'font-size':font_size_text, 'fill':'black', \
                'font-family':fontname, 'text-anchor':'start'}
            sampleline = inkex.etree.SubElement(txt_grp, inkex.addNS('text', 'svg'), text_attribs)
            sampleline.set('id', self.id_new())
            sampleline.text = self.options.sample_text
            sampleline.set('style', simplestyle.formatStyle(sampletext_style))
            y_current += y_offset
        self.recursively_traverse_svg(txt_grp, self.doc_transform)


    def glyph_table(self):
        """
        Generate display table of glyphs within the current SVG font.
        Sorted display of all printable characters in the font _except_ missing glyph.
        """

        fontname = self.font_load_wrapper('not_a_font_name') # force load of default

        if self.font_load_fail:
            inkex.errormsg('Font not found; Unable to generate glyph table.')
            return

        txt_grp = inkex.etree.SubElement(self.current_layer, 'g') # Embed text in a group
        txt_grp.set('id', self.id_new())

        glyph_count = 0
        for glyph in self.font_dict[fontname]['glyphs']:
            if self.font_dict[fontname]['glyphs'][glyph]['d'] is not None:
                glyph_count += 1

        columns = int(math.floor(math.sqrt(glyph_count)))

        font_size_temp = 0.4 # in inches; will be scaled by viewbox factor.
        font_size_text = str(font_size_temp / self.vb_scale) + 'px'
        glyph_style = simplestyle.formatStyle({'stroke' : 'none',\
                'font-size':font_size_text, 'fill' : 'black', \
                'font-family' : fontname, 'text-anchor': 'start'})
        x_offset = 1.5 * font_size_temp / self.vb_scale
        y_offset = x_offset
        draw_position = 0

        for glyph in sorted(self.font_dict[fontname]['glyphs']):
            if self.font_dict[fontname]['glyphs'][glyph]['d'] is None:
                continue
            y_pos, x_pos = divmod(draw_position, columns)
            text_attribs = {'x':str(x_offset * (x_pos + 1)), 'y': str(y_offset * (y_pos + 1))}
            sampleline = inkex.etree.SubElement(txt_grp, inkex.addNS('text', 'svg'), text_attribs)
            sampleline.set('id', self.id_new())
            sampleline.text = glyph
            sampleline.set('style', glyph_style)
            draw_position = draw_position + 1

        self.recursively_traverse_svg(txt_grp, self.doc_transform)


    def find_font_files(self):
        '''
        Create list of "plausible" SVG font files

        List items in primary svg_fonts directory, typically located in the
        directory where this script is being executed from.

        If there is text given in the "Other SVG Font" input, that text may
        represent one of the following:

        (A) The name of a font file, located in the svg_fonts directory.
            - This may be given with or without the .svg suffix.
            - If it is a font file, and the font face selected is "other",
                then use this as the default font face.

        (A2) The name of a font file, located in the svg_fonts directory
            was entered, but Inkscape may "helpfully" add a full path to it.

        (B) The path to a font file, located elsewhere.
            - If it is a font file, and the font face selected is "other",
                then use this as the default font face.
            - ALSO: Search the directory where that file is located for other SVG fonts.

        (C) The path to a directory
            - It may or may not have a trailing separator
            - Search that directory for SVG fonts.

        This function will create a list of available files that appear to be SVG (SVG font)
        files. It does not parse the files. We will format it as a dictionary that maps each
        file name (without extension) to a path.

        External fonts (those listed in an external "svg_fonts" directory)
        take priority over fonts in the built-in svg_fonts directory.
        '''

        def add_fonts(font_dir):
            ''' add all svg files inside `font_dir` to `self.font_file_list`, overriding
            already-existing entries if there are key conflicts '''

            for dir_item in os.listdir(font_dir):
                if dir_item.endswith((".svg", ".SVG")):
                    file_path = os.path.join(font_dir, dir_item)
                    if os.path.isfile(file_path): # i.e., if not a directory
                        root, _ext = os.path.splitext(dir_item)
                        self.font_file_list[root.lower()] = file_path

        self.font_file_list = dict()

        # List contents of primary font directory:
        # Try a few approaches to locating the font directory:
        font_dirs = [
            # List of possible locations for the font directory, in reverse order of priority
            os.path.abspath(os.path.dirname(__file__)), # Directory where this file is located
            os.path.dirname(os.path.realpath(sys.argv[0])), # Directory where this was called
            os.getcwd(), # Try the current working directory
            ]

        # walk through the font_dirs list, which is in reverse order of priority, so high
        # priority fonts override low priority fonts in the font_file_list dict
        font_directory_name = 'svg_fonts'
        for font_dir in font_dirs:
            full_path = os.path.realpath(os.path.join(font_dir, font_directory_name))
            if os.path.exists(full_path):
                add_fonts(full_path)

        # Check for case "(B)": A file, not in primary font directory
        test_path = os.path.realpath(os.path.expanduser(self.options.font_face))
        if os.path.isfile(test_path):
            directory, file_name = os.path.split(test_path)
            root, _ext = os.path.splitext(file_name)
            self.font_file_list[root.lower()] = test_path

            if self.options.font_option is None:
                self.options.font_option = root
            elif self.options.font_option == "other":
                self.options.font_option = root

            # Also search the directory where that file
            # was located for other SVG files (which may be fonts)
            add_fonts(directory)
            return

        # Check for case "(C)": A directory name
        if os.path.isdir(test_path):
            add_fonts(test_path)

        # split off file extension (e.g., ".svg") from "Other font" input
        root, _ext = os.path.splitext(self.options.font_face)

        # Case "(A)": Input text is the name of an item in the primary font directory.
        if root.lower() in self.font_file_list:
            # If we already have that name in our font_file_list, and None or "other" is
            # selected, this is now our default font face.

            if self.options.font_option is None:
                self.options.font_option = root.lower()
                return

            if self.options.font_option == "other":
                self.options.font_option = root.lower()
            return
            # If the value entered in the "Other font" input is the name of a
            # font (rather than the path to an external font or font directory),
            # then return because here is no external directory to search.

        root, _ext = os.path.splitext(os.path.basename(self.options.font_face))

        # Case "(A2)": Input text is the name of an item in the primary font directory,
        # with extraneous path prepended. This can happen if you enter "myfont.svg" in the
        # "font_face" box without a full path.
        if root.lower() in self.font_file_list:
            # If we already have that name in our font_file_list, and None or "other" is
            # selected, this is now our default font face.

            if self.options.font_option is None:
                self.options.font_option = root.lower()
                return

            if self.options.font_option == "other":
                self.options.font_option = root.lower()
            return
            # If the value entered in the "Other font" input is the name of a
            # font (rather than the path to an external font or font directory),
            # then return because here is no external directory to search.


    def font_load_wrapper(self, fontname):
        '''
        Implements the following logic for loading fonts:

        * Check to see if the font name is in our lookup table of fonts,
            self.font_dict

        * If the font is not listed in font_dict[]:
            * Check to see if there is a corresponding SVG font file that
            can be opened and parsed.

            * If the font can be opened and parsed:
                * Add that font to font_dict.
            * Otherwise
                * Add the font name to font_dict as None.

        * If the font has value None in font_dict:
            * Try to load fallback font.

        * Fallback font:
            * If an SVG font matching that in the SVG is not available,
            check to see if the default font is available. That font
            is given by self.options.font_option

            * Secondary fallback: If the main fallback is not available,
            check to see if the secondary default, given by
            self.params.font_option is available.

        * If a font is loaded and available, return the font name.
            Otherwise, return none.
        '''

        fontname = fontname.lower()
        # Check font face as specified in the document's styling
        #   This is how the font is matched for automatic font mapping
        self.load_font(fontname)

        if self.font_dict[fontname] is None:

            if self.options.font_option is None:
                self.font_load_fail = True # Set failure flag
                return None

            # If we were not able to load the requested font:
            fontname = self.options.font_option.lower()    # First fallback
            if DEBUG:
                inkex.errormsg('[get_font_char]: Try first fallback: ' + str(fontname))
            if fontname not in self.font_dict:
                self.load_font(fontname)
            else:
                pass

        if self.font_dict[fontname] is None:
            fontname = self.params.font_option.lower()    # Second fallback: Default font
            if DEBUG:
                inkex.errormsg('[get_font_char]: Try second fallback: ' + str(fontname))
            if fontname not in self.font_dict:
                self.load_font(fontname)

        if self.font_dict[fontname] is None:
            self.font_load_fail = True # Set a flag to only generate one copy of this error.
            return None
        return fontname


    def get_font_char(self, fontname, char):
        '''
        Given a font face name and a character (unicode point), return an SVG path,
            horizontal advance value, and scaling factor.

        If the font is not available by name, use the default font.
        '''

        fontname = self.font_load_wrapper(fontname) # Load the font if available

        if fontname is None:
            return None
        if not char:
            return None

        try:
            scale_factor = self.font_dict[fontname]['scale']
        except:
            scale_factor = 0.001  # Default: 1/1000

        try:
            if ord(char) == 8203: # Detect zero-width space character
                return None
            if char not in self.font_dict[fontname]['glyphs']:
                x_adv = self.font_dict[fontname]['missing_glyph']['horiz_adv_x']
                x_adv *= (self.options.letter_spacing / 100.0) # Convert from percent
                return self.font_dict[fontname]['missing_glyph']['d'], \
                    x_adv, scale_factor

            if DEBUG:
                inkex.errormsg('char to render: ' + \
                str(self.font_dict[fontname]['glyphs'][char]['glyph_name']))

            x_adv = self.font_dict[fontname]['glyphs'][char]['horiz_adv_x']
            if char.isspace() or char in ['Ἧ', 'ἰ']: # Several unicode points may represent spaces
                x_adv *= self.options.word_spacing / 100.0 # Convert from percent
            else:
                x_adv *= self.options.letter_spacing / 100.0 # Convert from percent

            return self.font_dict[fontname]['glyphs'][char]['d'], \
                x_adv, scale_factor
        except:
            return None


    def handle_viewbox(self):
        """ Initial transform of document is based on viewbox, if present """
        self.svg_height = plot_utils.getLengthInches(self, 'height')
        self.svg_width = plot_utils.getLengthInches(self, 'width')

        self.svg = self.document.getroot()
        v_b = self.svg.get('viewBox')
        if v_b:
            p_a_r = self.svg.get('preserveAspectRatio')
            s_x, s_y, o_x, o_y = plot_utils.vb_scale(v_b, p_a_r, self.svg_width, self.svg_height)
        else:
            s_x = 1.0 / float(plot_utils.PX_PER_INCH) # Handle case of no viewbox
            s_y = s_x
            o_x = 0.0
            o_y = 0.0

        self.doc_transform = simpletransform.parseTransform(
            'scale({0:.6E},{1:.6E}) translate({2:.6E},{3:.6E})'.format(s_x, s_y, o_x, o_y))

        # In case of non-square aspect ratio, use average value.
        self.vb_scale = (s_x + s_y) / 2.0

    def draw_svg_text(self, chardata, parent):
        """Draw individual glyph as an SVG path in the document"""
        char = chardata['char']

        if char in ('\n', '\r'):
            return 0, True  # No character to draw for new_line

        font_family = chardata['font_family']
        offset = chardata['offset']
        vertoffset = chardata['vertoffset']
        font_height = chardata['font_size']
        scale = 1.0

        # Stroke scale factor, including external transformations:
        stroke_scale = chardata['stroke_scale'] * self.vb_scale

        try:
            path_string, adv_x, scale_factor = self.get_font_char(font_family, char)
        except:
            adv_x = 0
            path_string = None
            scale_factor = 1.0

        if self.font_load_fail:
            return 0, True

        scale *= scale_factor * font_height
        h_offset = 0
        v_offset = 0
        trans = ""

        if self.options.enable_defects:
            if self.options.baseline_var > 0:
                baseline_local = (self.s_rand.random() - 0.5) * self.options.baseline_var / 500.0
                if (self.baseline_offset > self.max_baseline_var) and (baseline_local > 0):
                    baseline_local = -baseline_local    #steer towards zero
                if (self.baseline_offset < -self.max_baseline_var) and (baseline_local < 0):
                    baseline_local = -baseline_local    #steer towards zero
                self.baseline_offset += baseline_local
                v_offset = self.baseline_offset * font_height

            if self.options.indent_var > 0:
                if self.new_line:
                    self.new_line = False
                    indent_temp = -1.0
                    while (indent_temp < 0) or (indent_temp > self.max_indent_var):
                        indent_temp = self.indent_offset + (self.s_rand.random() - 0.5) *\
                            self.options.indent_var / 50.0
                    self.indent_offset = indent_temp
                    self.line_offset = self.indent_offset * font_height
                    self.half_max_indent = font_height * self.max_indent_var / 2

            if self.options.kern_var > 0:
                kern_local = (self.s_rand.random() - 0.5) * self.options.kern_var / 200.0
                if (self.kern_offset > (self.max_kern_var)) and (kern_local > 0):
                    kern_local = -kern_local    #steer towards zero
                if (self.kern_offset < -self.max_kern_var) and (kern_local < 0):
                    kern_local = -kern_local    #steer towards zero
                self.kern_offset += kern_local
                h_offset += self.kern_offset * float(adv_x) * scale

            size_var = self.options.size_var / 200.0 # Value is in the range (0, 0.5).
            if size_var > 0:
                # If self.max_size_var is 1.0, then self.fontsize_offset can be up in the range
                #   (-0.5, 0.5) before being steered back.
                size_local = (self.s_rand.random() - 0.5) * size_var * 0.25
                # The actual scale value range is then: (0.5, 1.5)
                # This can, of course, be dialed back with the control parameter.
                if (self.fontsize_offset > (self.max_size_var * size_var)) and (size_local > 0):
                    size_local = -size_local    # Steer towards zero
                if (self.fontsize_offset < -(self.max_size_var * size_var)) and (size_local < 0):
                    size_local = -size_local    # Steer towards zero
                self.fontsize_offset += size_local
                scale = scale * (1 + self.fontsize_offset)

        # SVG fonts use inverted Y axis; mirror vertically
        scale_text = 'scale('+format(scale, '.6f')+', -'+format(scale, '.6f')+')'

        # Combine scales of external transformations with the scaling applied by this function:
        stroke_width = self.render_width / (scale * stroke_scale)

        # Stroke-width is a css style element; cannot use scientific notation.
        # Use variable precision to encoding the stroke width:
        log_ten = math.log10(stroke_width)
        if log_ten > 0:  # For stroke_width > 1
            width_string = "{0:.3f}in".format(stroke_width)
        else:
            prec = int(math.ceil(-log_ten) + 3)
            width_string = "{0:.{1}f}in".format(stroke_width, prec)
        p_style = {'stroke-width': width_string}


        trans += 'translate(' + format(offset + h_offset, '.6f') + ',' +\
            format(vertoffset + v_offset, '.6f') + ')'
        trans += scale_text
        text_attribs = {'d':path_string, 'transform':trans,\
                                        'style': simplestyle.formatStyle(p_style)}

        if path_string is not None:
            _path_ref = inkex.etree.SubElement(parent, inkex.addNS('path', 'svg'), text_attribs)
            _path_ref.set('id', self.id_new())
            self.output_generated = True

        # Return (new horizontal offset value, True if character is whitespace):
        return offset + float(adv_x) * scale, path_string is None

    def call_substitution_api(self):
        """Scriptalizer text replacement with compatible fonts by Quantum Enterprises"""

        requests = from_dependency_import('requests')
        self.req_sess = requests.Session()
        api_called = False
        api_error = False
        for fontname in self.snippets: # Iterate over fonts awaiting substitution
            url = "https://www.scriptalizer.co.uk/scriptalizer.asmx/ScriptalizeByFont"

            payload = {'FontName': fontname, 'ErrFrequency': self.params.script_mistakes,
                'InputText': self.snippets[fontname]}
            if DEBUG:
                inkex.debug("API payload: " + str(payload))

            api_called = True
            self.api_calls = self.api_calls + 1
            api_response = None
            try:
                if DEBUG:
                    inkex.debug("Trying scriptalizer: ")
                api_response = self.req_sess.post(url,\
                    data=payload, timeout=self.params.script_timeout)

                if DEBUG:
                    inkex.debug(f"Response time(s): {api_response.elapsed.total_seconds():.3f}")

            except:
                api_error = True
                if not self.params.script_quiet:
                    inkex.debug("An error occurred while Scriptalizing text." +
                        " Check your internet connection.")
                    self.error_code = 100
                return

            if not api_response:
                api_error = True
                if DEBUG or not self.params.script_quiet:
                    inkex.debug("An error occurred while Scriptalizing text.")
                    self.error_code = 101
                return
            if api_response.status_code != 200:
                api_error = True
                if not self.params.script_quiet:
                    self.error_code = 102
                    inkex.debug("Scriptalizer error.")
                if DEBUG:
                    inkex.debug(api_response)
                return
            if api_response.text:
                if DEBUG:
                    inkex.debug(api_response.text.encode('utf-8'))
                root = etree.fromstring(api_response.text.encode('utf-8'))
                status = None
                output = None

                for node in root:
                    tag = etree.QName(node)
                    if tag.localname == "Status":
                        status = node.text.encode('utf-8')
                    if tag.localname == "OutputText":
                        output = node.text

                if status != "OK".encode('utf-8'):
                    api_error = True
                    if DEBUG or not self.params.script_quiet:
                        inkex.debug("Scriptalizer returned an error: " + str(status))
                        self.error_code = 103
                    return
                if not output:
                    return

                output = output[:-1] # An extra space is added on the end.
                # Last two chars: LF, space -- ascii 10, 32.
                self.snippets[fontname] = output

        if api_called and not api_error:
            self.substitution_performed = True


    def text_replace_read(self, font_name, text):
        ''' Add text elements to delimited unicode string, one string per font face '''

        # Identify currently-active font
        svg_font = self.font_load_wrapper(font_name)

        if svg_font is None:
            return # No SVG font available

        if not self.font_dict[svg_font]['qes']:
            return # Current font does not support scriptalizer

        # If font is scriptalizable, add the text to the snippet list for that font:
        if svg_font in self.snippets:
            self.snippets[svg_font] += self.text_delimiter + text
        else:
            self.snippets[svg_font] = self.text_empty + text


    def text_replace_write(self, font_name):
        ''' Replace text elements from delimited unicode string, one string per font face '''

        svg_font = self.font_load_wrapper(font_name) # Identify currently-active font
        if svg_font is None:
            return None # No SVG font available
        if not self.font_dict[svg_font]['qes']:
            return None # Current font does not support scriptalizer
        if (svg_font in self.snippets) and (svg_font in self.snip_index):
            snippet_array = self.snippets[svg_font].split(self.text_delimiter)
            output = None
            try:
                output = snippet_array[self.snip_index[svg_font]]
            except:
                if not self.params.script_quiet:
                    inkex.debug("Error splitting Scriptalizer output text; possible API error")
            self.snip_index[svg_font] += 1
            return output
        return None


    def text_substitution(self, the_node_list, revert_mode=False):
        '''
        Perform Scriptalizer text substitution through the document
        where applicable.

        Method:
            * Recursively parse the SVG document, looking for text elements
            * Build the text elements into a long single unicode string,
                with an appropriate delimiter: '( =^..^= )'.
            * Use a separate unicode string for each font
            * Call the Scriptalizer API _once_ for each compatible font.
            * Recursively parse the SVG document in the same way, now
                replacing the text in the elements

        If called _after_ rendering text into SVG fonts with revert_mode=True
        _and_ we are preserving original text, then also revert text to the
        original text, using the same snippet-based replacement method.
        '''

        if not self.params.script_enable:
            return # Scriptalizer disabled in user preferences

        if not self.options.preserve_text:
            self.revert_script_text = False

        if not revert_mode: # Prepare initial substitution
            # text_snippets dictionary - One unicode-string per font face
            self.snippets = dict()

            # Compile dictionary of text snippets per font:
            self.text_replace_mode = 'read'       # Read mode: Populate dictionaries
            self.svg_preparse(the_node_list)

            if self.revert_script_text: # If we are planning to revert text after API call...
                self.backup_snippets = deepcopy(self.snippets) # then make a backup
            self.call_substitution_api() # Perform API call

            if not self.substitution_performed: # in case of API failure, do no substitution
                return
            self.snip_index = dict()
            for key in self.snippets:
                self.snip_index[key] = 0

            self.text_replace_mode = 'write'        # Write mode: Replace text in place
            self.svg_preparse(the_node_list)
        else:  # Undo substitution already performed
            if not self.revert_script_text:
                return
            if not self.substitution_performed:
                return
            self.snip_index = dict()
            for key in self.snippets:
                self.snip_index[key] = 0

            self.snippets = deepcopy(self.backup_snippets) # restore from backup
            self.text_replace_mode = 'write'        # Write mode: Replace text in place
            self.svg_preparse(the_node_list)


    def svg_preparse(self, the_node_list, parent_visibility='visible'):
        """
        Recursive routine to pre-parse the document and use Scriptalizer API
        to perform text substitution where applicable.
        """

        for node in the_node_list:
            vis = node.get('visibility', parent_visibility) # Ignore invisible nodes
            if vis == 'inherit':
                vis = parent_visibility
            if vis in ('hidden', 'collapse'):
                continue

            font_family = 'sans-serif' #default

            # First apply the current matrix transform to this node's tranform
            if node.tag == inkex.addNS('g', 'svg') or node.tag == 'g':
                recurse_group = True
                ink_label = node.get(inkex.addNS('label', 'inkscape'))

                if not ink_label:
                    pass
                else:
                    if ink_label == 'hta':
                        recurse_group = False    # Do not traverse groups of rendered text.
                if recurse_group:
                    self.svg_preparse(node, vis)

            elif node.tag == inkex.addNS('use', 'svg') or node.tag == 'use':
                continue # Ignore cloned nodes in this preparsing

            elif (node.tag == inkex.addNS('text', 'svg')) or (node.tag == 'text') or\
                                    (node.tag == inkex.addNS("flowRoot", "svg")):
                hershey_ignore = node.get('hta-ignore')
                if hershey_ignore is not None:
                    continue # If the attribute is present, skip this node.

                try:
                    # TODO: Revise this; using proper inheritance for font family
                    node_style = simplestyle.parseStyle(node.get('style'))
                    font_family = hta_utils.strip_quotes(node_style['font-family'])
                except:
                    pass

                if node.tag == inkex.addNS("flowRoot", "svg"):
                    the_id = node.get('id') #CASE: SVG 1.2 style Flowed text

                    #selects the flowRegion's child (svg:rect) to get @X and @Y
                    flowref = self.xpathSingle('/svg:svg//*[@id="%s"]/svg:flowRegion[1]'\
                        % the_id)[0]
                    if flowref.tag != inkex.addNS("rect", "svg"):
                        continue # Handle only case of flowed text in a rectangle.

                    self.flowroot_preparse(node, font_family)

                else:    # If this is a text object, rather than a flowroot object:
                    # CASE: Regular (non-flowroot) text node
                    self.text_preparse(node, font_family)


    def flowroot_preparse(self, node_list, font_family):
        """Initial pass through flowroot object"""
        font_local = font_family

        for node in node_list:
            try:
                node_style = simplestyle.parseStyle(node.get('style'))
                font_local = hta_utils.strip_quotes(node_style['font-family'])
            except:
                pass

            if node.text is not None:
                node.text = node.text.replace('\\r', '🐍')

                if self.text_replace_mode == 'write': # Write mode
                    substitute = self.text_replace_write(font_local)
                    if substitute is not None:
                        node.text = substitute
                else:   # Read mode
                    self.text_replace_read(font_local, node.text)

            if ((node.tag == inkex.addNS("flowPara", "svg")) or\
                                (node.tag == inkex.addNS("flowSpan", "svg")) or\
                                (node.tag == 'flowPara') or (node.tag == 'flowSpan')):
                self.flowroot_preparse(node, font_local)

            if node.tail is not None:
                tail = node.tail
                if tail is not None:
                    font_local = font_family
                    tail = tail.replace('\\r', '🐍')

                    if self.text_replace_mode == 'write': # Write mode
                        substitute = self.text_replace_write(font_local)
                        if substitute is not None:
                            node.tail = substitute
                    else:   # Read mode
                        self.text_replace_read(font_local, tail)


    def text_preparse(self, node, font_family):
        """Initial pass through text object"""
        font_local = font_family

        try:
            node_style = simplestyle.parseStyle(node.get('style'))
        except:
            pass
        try:
            font_local = hta_utils.strip_quotes(node_style['font-family'])
        except:
            pass

        if node.text is not None:
            node.text = node.text.replace('\\r', '🐍')

            if self.text_replace_mode == 'write': # Write mode
                substitute = self.text_replace_write(font_local)
                if substitute is not None:
                    node.text = substitute
            else:   # Read mode
                self.text_replace_read(font_local, node.text)

        for subnode in node:
            if ((subnode.tag == inkex.addNS('tspan', 'svg')) or (subnode.tag == 'tspan')):
                self.text_preparse(subnode, font_local)

        if node.tail is not None:
            tail = node.tail
            if tail is not None:
                font_local = font_family
                tail = tail.replace('\\r', '🐍')
                if self.text_replace_mode == 'write': # Write mode
                    substitute = self.text_replace_write(font_local)
                    if substitute is not None:
                        node.tail = substitute
                else:   # Read mode
                    self.text_replace_read(font_local, tail)


    def recursive_parse_flowroot(self, node_list, parent_styles):
        '''
        Search through flowroot and child elements
        parent_styles: A dict of text-relevant styles and values
        '''

        if DEBUG:
            inkex.errormsg("recursive_parse_flowroot: " )

        for node in node_list:

            node_style_parsed = simplestyle.parseStyle(node.get('style'))
            node_style_dict = hta_utils.inherit_styles(parent_styles, node_style_parsed, False)

            font_size_local = node_style_dict['font-size']
            font_family_local = node_style_dict['font-family']
            line_spacing_local = node_style_dict['line-height']
            text_align_local = node_style_dict['text-align']
            para_size_local = node_style_dict['font-size-paragraph']
            para_height_local = node_style_dict['line-height-paragraph']

            undef_local = node_style_parsed.get("line-height") is None

            if node.text is not None:
                node.text = node.text.replace('🐍', '\r')
                self.text_string += node.text

                for _char in node.text:
                    self.text_families.append(font_family_local)
                    self.text_heights.append(font_size_local)
                    self.text_space.append(line_spacing_local)
                    self.text_aligns.append(text_align_local)
                    self.para_size.append(para_size_local)
                    self.para_height.append(para_height_local)
                    self.undefined_height.append(undef_local)

            if ((node.tag == inkex.addNS("flowPara", "svg")) or\
                    (node.tag == inkex.addNS("flowSpan", "svg")) or\
                    (node.tag == 'flowPara') or (node.tag == 'flowSpan')):
                self.recursive_parse_flowroot(node, node_style_dict)

            if node.tail is not None:
                node.tail = node.tail.replace('🐍', '\r')

                font_size_local = parent_styles['font-size']  # By default, inherit from parent.
                font_family_local = parent_styles['font-family']
                line_spacing_local = parent_styles['line-height']
                text_align_local = parent_styles['text-align']
                para_size_local = node_style_dict['font-size-paragraph']
                para_height_local = node_style_dict['line-height-paragraph']

                self.text_string += node.tail
                for _char in node.tail:
                    self.text_families.append(font_family_local)
                    self.text_heights.append(font_size_local)
                    self.text_space.append(line_spacing_local)
                    self.text_aligns.append(text_align_local)
                    self.para_size.append(para_size_local)
                    self.para_height.append(para_height_local)
                    self.undefined_height.append(undef_local)

            if node.tag == inkex.addNS("flowPara", "svg"):
                self.text_string += "\n"    # Conclude every flowpara with a return
                self.text_families.append(font_family_local)
                self.text_heights.append(font_size_local)
                self.text_space.append(line_spacing_local)
                self.text_aligns.append(text_align_local)
                self.para_size.append(para_size_local)
                self.para_height.append(para_height_local)
                self.undefined_height.append(undef_local)

    def parse_svg2_flowtext(self, node_list, parent_styles):
        """ Recursively parse svg 2 style flowed text """
        if DEBUG:
            inkex.errormsg('parse_svg2_flowtext: ' )

        for node in node_list:

            node_style_parsed = simplestyle.parseStyle(node.get('style'))
            node_style_dict = hta_utils.inherit_styles(parent_styles, node_style_parsed, False)

            font_size_local = node_style_dict['font-size']
            font_family_local = node_style_dict['font-family']
            line_spacing_local = node_style_dict['line-height']
            text_align_local = node_style_dict['text-align']
            para_size_local = node_style_dict['font-size-paragraph']
            para_height_local = node_style_dict['line-height-paragraph']

            if node.text is not None:
                node.text = node.text.replace('🐍', '\r')
                self.text_string += node.text

                for _char in node.text:
                    self.text_families.append(font_family_local)
                    self.text_heights.append(font_size_local)
                    self.text_space.append(line_spacing_local)
                    self.text_aligns.append(text_align_local)
                    self.para_size.append(para_size_local)
                    self.para_height.append(para_height_local)

            if ((node.tag == inkex.addNS('tspan', 'svg')) or (node.tag == 'tspan')):
                self.recursive_parse_flowroot(node, node_style_dict)

            if node.tail is not None:
                node.tail = node.tail.replace('🐍', '\r')

                font_size_local = parent_styles['font-size']  # By default, inherit from parent.
                font_family_local = parent_styles['font-family']
                line_spacing_local = parent_styles['line-height']
                text_align_local = parent_styles['text-align']
                para_size_local = node_style_dict['font-size-paragraph']
                para_height_local = node_style_dict['line-height-paragraph']

                self.text_string += node.tail
                for _char in node.tail:
                    self.text_families.append(font_family_local)
                    self.text_heights.append(font_size_local)
                    self.text_space.append(line_spacing_local)
                    self.text_aligns.append(text_align_local)
                    self.para_size.append(para_size_local)
                    self.para_height.append(para_height_local)

    def recursive_parse_textnode(self, node, parent_info, parent_styles, level):
        """Recursively parse text nodes and children"""
        if DEBUG:
            inkex.errormsg('recursive_parse_textnode: ' )

        x_local = parent_info['x_pos']
        y_local = parent_info['y_pos']

        font_size_local = parent_styles['font-size'] # By default, inherit values from parent.
        font_family_local = parent_styles['font-family']
        anchor_local = parent_styles['text-anchor']
        parent_line_spacing = parent_styles['line-height']

        if (node.tag == inkex.addNS('text', 'svg')) or (node.tag == 'text'):
            self.y_cr_offset = 0.0 # Keep track of manually added carriage returns

        if level == 0:
            # Called from recursively_traverse_svg; not recursive_parse_textnode
            node_style_dict = parent_styles.copy() # Already parsed at this nesting level.
        else:
            node_style_parsed = simplestyle.parseStyle(node.get('style'))
            node_style_dict = hta_utils.inherit_styles(parent_styles, node_style_parsed, False)

        font_size_local = node_style_dict['font-size']
        font_family_local = node_style_dict['font-family']
        anchor_local = node_style_dict['text-anchor']

        x_temp = node.get('x')
        if x_temp is not None:
            try:
                x_local = float(x_temp)
            except ValueError:
                self.warn_unkern = True
                return

        y_spacing = hta_utils.line_space_calc(font_size_local,
            parent_line_spacing)

        adv_line = False
        role = node.get(inkex.addNS('role', 'sodipodi'))
        if role == "line":
            adv_line = True

        y_temp = node.get('y')
        if y_temp is not None:
            y_local = float(y_temp)
        else:
            role = node.get(inkex.addNS('role', 'sodipodi'))
            if role == "line": # tspan represents a new line, but no y value given:
                y_local = float(y_local) + self.line_number * y_spacing
            elif y_local is not None:
                y_local = float(y_local)

        if node.text is not None:
            node.text = node.text.replace('🐍', '\r')

            self.text_string += node.text

            for char in node.text:
                self.text_families.append(font_family_local)
                self.text_heights.append(font_size_local)
                self.text_aligns.append(anchor_local)
                self.text_x.append(x_local)
                if y_local is not None: # If text is on a path, typically
                    self.text_y.append(y_local + self.y_cr_offset)
                if char == '\r':
                    self.y_cr_offset += y_spacing # add spacing *after* '\r' character

        for subnode in node:
            # If text is located within a subnode of this node, process that subnode

            if ((subnode.tag == inkex.addNS('textPath', 'svg')) or (subnode.tag == 'textPath')):
                self.warn_text_path = True
                continue

            if ((subnode.tag == inkex.addNS('tspan', 'svg')) or (subnode.tag == 'tspan')):
                # Note: Possibly add additional text tags in the future?

                node_info = {'x_pos': x_local, 'y_pos': y_local}

                adv_line = False
                role = subnode.get(inkex.addNS('role', 'sodipodi'))
                if role == "line":
                    adv_line = True

                self.recursive_parse_textnode(subnode, node_info, node_style_dict, 1)

                if adv_line: # Increment line after tspan if it is labeled as a line
                    self.line_number = self.line_number + 1

        if node.tail is not None:
            _stripped_tail = node.tail.strip()
            if _stripped_tail is not None:

                font_size_local = parent_styles['font-size']    # By default, inherit from parent.
                font_family_local = parent_styles['font-family']
                text_align_local = parent_styles['text-anchor']

                x_local = parent_info['x_pos']
                y_local = parent_info['y_pos']
                self.text_string += _stripped_tail
                for char in _stripped_tail:
                    self.text_heights.append(font_size_local)
                    self.text_families.append(font_family_local)
                    self.text_aligns.append(text_align_local)
                    self.text_x.append(x_local)
                    self.text_y.append(y_local)


    def recursively_traverse_svg(self, the_node_list, mat_current=None,
                                 parent_visibility='visible', parent_styles=None):
        """Recursively parse the SVG to perform font substitution"""

        if mat_current is None:
            mat_current = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        for node in the_node_list:
            vis = node.get('visibility', parent_visibility) # Ignore invisible nodes
            if vis == 'inherit':
                vis = parent_visibility
            if vis in ('hidden', 'collapse'):
                continue

            locked_var = node.get(inkex.addNS('insensitive', 'sodipodi'))

            if locked_var == "true":
                continue # Object is locked; skip

            node_style_parsed = simplestyle.parseStyle(node.get('style'))

            # Check for "display:none" in the node's style attribute:
            if 'display' in node_style_parsed.keys() and node_style_parsed['display'] == 'none':
                continue  # Do not parse this object or its children
            # The node may have a display="none" attribute as well:
            if node.get('display') == 'none':
                continue  # Do not parse this object or its children

            # Apply the current matrix transform to this node's tranform
            mat_new = simpletransform.composeTransform(mat_current,
                simpletransform.parseTransform(node.get("transform")))

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

                node_style_dict = hta_utils.inherit_styles(parent_styles, node_style_parsed)
                self.recursively_traverse_svg(node, mat_new, vis, node_style_dict)

            elif node.tag == inkex.addNS('use', 'svg') or node.tag == 'use':

                # A <use> element refers to another SVG element via an xlink:href="#blah"
                # attribute.  We will handle the element by doing an XPath search through
                # the document, looking for the element with the matching id="blah"
                # attribute.  We then recursively process that element after applying
                # any necessary (x,y) translation.
                #
                # Notes:
                #  1. We ignore the height and width attributes as they do not apply to
                #     path-like elements, and
                #  2. Even if the use element has visibility="hidden", SVG still calls
                #     for processing the referenced element.  The referenced element is
                #     hidden only if its visibility is "inherit" or "hidden".

                refid = node.get(inkex.addNS('href', 'xlink'))
                if not refid:
                    continue # missing reference

                path = '//*[@id="%s"]' % refid[1:] # # [1:] to ignore leading '#' in reference
                refnode = node.xpath(path)
                if refnode:
                    local_transform = simpletransform.parseTransform(node.get("transform"))
                    x_val = float(node.get('x', '0'))
                    y_val = float(node.get('y', '0'))

                    if x_val != 0 or y_val != 0:
                        local_transform2 = simpletransform.composeTransform(local_transform,\
                        simpletransform.parseTransform(\
                        'translate({0:.6E},{1:.6E})'.format(x_val, y_val)))
                    else:
                        local_transform2 = local_transform

                    try:
                        ref_group = inkex.etree.SubElement(the_node_list, 'g') # Add a subgroup
                        the_id = self.id_new()
                        ref_group.set('id', the_id)
                    except TypeError:
                        inkex.errormsg('Unable to process selected nodes. ' +
                            'Consider unlinking cloned text.')
                        continue

                    the_id = ref_group.get('id')
                    if not the_id:
                        the_id = self.id_new()
                        ref_group.set('id', the_id)

                    ref_group.set('transform', simpletransform.formatTransform(local_transform2))

                    id_list = []

                    for subnode in refnode:
                        the_id = subnode.get('id')
                        if not the_id:
                            the_id = self.id_new()
                            subnode.set('id', the_id)

                        if the_id not in id_list:
                            ref_group.append(deepcopy(subnode)) # add node at end of node_list
                            id_list.append(the_id)

                    for subnode in ref_group:
                        # The copied text elements should be removed at the end,
                        # or they will persist if original elements are preserved.
                        self.nodes_to_delete.append(subnode)

                    #Preserve original element?
                    if not self.options.preserve_text:
                        self.nodes_to_delete.append(node)

            if node.tag not in (inkex.addNS('text', 'svg'), 'text',
                    inkex.addNS("flowRoot", "svg")):
                continue

            node_style_dict = hta_utils.inherit_styles(parent_styles, node_style_parsed)

            # Variables are initially zeroed for each text object.
            self.baseline_offset = 0.0    # Baseline Shift
            self.kern_offset = 0.0
            self.fontsize_offset = 0.0    # Deviation of font size away from nominal
            self.new_line = True # Flag new line of text, for calculating new indent
            self.indent_offset = 0

            start_x = 0  # Defaults; Fail gracefully in case xy position is not given.
            start_y = 0

            start_x = node.get('x')    # XY Position of element
            start_y = node.get('y')

            bounding_rect = False
            rect_width = 100         #default size of bounding rectangle for flowroot object
            transform = ""          #transform (scale, translate, matrix, etc.)
            text_align = "start"
            first_line = True # For first-line text height, and center-aligned text

            hershey_ignore = node.get('hta-ignore')
            if hershey_ignore is not None:
                continue # If the attribute is present, skip this node.
            transform = node.get('transform')

            if transform is not None:
                transform2 = simpletransform.parseTransform(transform)
                '''
                Compute estimate of transformation scale applied to this element, for purposes
                of calculating the stroke width to apply. When all transforms are applied and
                our elements are displayed, we want the visible stroke width to be reasonable.
                Transformation matrix is [[a c e][b d f]]
                scale_x = sqrt(a * a + b * b),
                scale_y = sqrt(c * c + d * d)
                Take estimated scale as the mean of the two.
                '''
                scale_x = math.sqrt(transform2[0][0] * transform2[0][0] +
                                    transform2[1][0] * transform2[1][0])
                scale_y = math.sqrt(transform2[0][1] * transform2[0][1] +
                                    transform2[1][1] * transform2[1][1])
                scale_r = (scale_x + scale_y) / 2.0 # Average.
            else:
                scale_r = 1.0

            the_id = node.get('id')     # Needed for flowref child identification

            #Initialize text attribute lists for each top-level text object:
            self.text_string = "" # Python 3 string (unicode-based)
            self.text_families = []   # List of font family for characters in the string
            self.text_heights = []    # List of font heights
            self.text_space = []      # List of vertical line heights
            self.text_aligns = []     # List of horizontal alignment values
            self.para_size = []       # List of paragraph-level font size style
            self.para_height = []     # List of paragraph-level line-height style
            self.undefined_height = [] # List booleans; for noting missing line-height

            self.text_x = []        # List; x-coordinate of text line start
            self.text_y = []        # List; y-coordinate of text line start

            # Group generated paths together, to make the rendered letters
            # easier to manipulate in Inkscape once generated:
            g_attribs = {inkex.addNS('label', 'inkscape'):'hta'}
            parent = node.getparent()

            self.output_generated = False

            text_block = inkex.etree.Element('g', g_attribs)
            pos = list(parent).index(node)
            parent.insert(pos, text_block) # insert in same position as text; preserve z-index

            style = {'stroke' : '#000000', 'fill' : 'none', \
                'stroke-linecap' : 'round', 'stroke-linejoin' : 'round'}

            # Apply rounding to ends to improve final engraved text appearance.
            text_block.set('style', simplestyle.formatStyle(style))

            # Some common variables used in any type of text:
            str_pos = 0      # Position through the full string that we are rendering
            i = 0            # Dummy (index) variable for looping over letters in string
            width_this_line = 0 # Estimated width of characters to be stored on this line
            strip_list = [] # List for holding strings prior to adding to their group

            """
            Determine method of text wrapping, if any, to use before proceeding,

            CASE A: SVG 1.2-style flowed text
                - Flowroot element, as opposed to a text element
                - Has sub-elements like flowpara & flowspan for paragraphs & spans
                - Defines flowRegion - a rectangle shape
            CASE B: SVG2-style flowed text, in specific cases that we can re-flow
                - Text element
                - style includes shape-inside, white-space:pre
                - shape-inside references an object by id _that exists_ in the document
                - Referenced object is a rect, with X, Y positions and width
                - user has self.params.reflow_svg2 enabled
                - If any of the above cases fail, we revert to case C.
            CASE C: Regular non-flowed text
                - Text element
                - Default handling of text nodes
                - Only place returns in designated places
                - Handles pre-formatted SVG2 wrapped text, if reflow option is disabled

            For cases A and B, compile a list of the text & styles in the text element,
            prior to wrapping text

            For case C, compile the list and perform no automatic wrapping.
            """

            text_handling = "non-flowed" # Default text handling: Case C
            reflow_textblock = False

            if node.tag == inkex.addNS("flowRoot", "svg"):
                text_handling = "flowed_svg1"   # Case A
                reflow_textblock = True
            elif self.params.reflow_svg2: # Text element; check for case B.
                shape_ref = node_style_parsed.get('shape-inside')
                if shape_ref:
                    shape_id = shape_ref[5:-1]
                    if shape_id:
                        path = '//*[@id="%s"]' % shape_id
                        refnode = self.xpathSingle( path )
                        if refnode is not None:
                            if refnode.tag == inkex.addNS("rect", "svg"):
                                start_x = refnode.get('x')
                                start_y = refnode.get('y')
                                rect_width = float(refnode.get('width'))
                                if start_x and start_y and rect_width:
                                    reflow_textblock = True # Case B
                                    text_handling = "flowed_svg2"

            if reflow_textblock: # Cases A, B
                text_align = node_style_dict['text-align']
                if text_handling == "flowed_svg1":
                    flowref = self.xpathSingle('/svg:svg//*[@id="%s"]/svg:flowRegion[1]'\
                        % the_id)[0]

                    # Currently support case of flowRegion pointing to rect, only.
                    # Adding support for a "use" element pointing to another rect
                    # is relatively simple, but does not seem to occur in real SVG files
                    if flowref.tag == inkex.addNS("rect", "svg"):
                        start_x = flowref.get('x', '0')
                        start_y = flowref.get('y', '0')
                        rect_width = float(flowref.get('width'))
                        if start_x and start_y and rect_width:
                            bounding_rect = True

                    if not bounding_rect:
                        # Skip the svg1 text block if its flow region is not well defined:
                        self.warn_unflow = True
                        continue

                    # Recursively loop through content of the flowroot object,
                    # looping through text, flowpara, and other things
                    # to build lists of text and style content:

                    self.recursive_parse_flowroot(node, node_style_dict)
                else: # i.e., when text_handling == "flowed_svg2":
                    self.parse_svg2_flowtext(node, node_style_dict)

                # After building lists of text and style content, loop through those lists,
                # one line at a time, finding how many words fit on a line, etc.
                if self.text_string == "" or self.text_string.isspace():
                    if not self.options.preserve_text:
                        self.nodes_to_delete.append(node)
                    continue # No convertible text in this SVG element.

                v = 0 # Initial vertical offset for the flowed text block:
                y_offs_overall = 0  #   critical for overall text block position.
                max_strip_width = 0

                # Split text by lines AND make a list of line lengths new_line characters.
                # We need to keep track of this to match up styles to printable glyphs.
                text_lines = self.text_string.splitlines()
                extd_text_lines = self.text_string.splitlines(True)
                str_pos_eol = 0 # str_pos after end of previous text_line.

                nbsp = '\xa0' # Unicode non-breaking space character
                for line_number, text_line in enumerate(text_lines):
                    line_length = len(text_line)
                    extd_line_length = len(extd_text_lines[line_number])
                    i = 0   # Position within this text_line.

                    # A text_line may take more than one strip to render.
                    line_start = 0 # Value of i when the current strip started.
                    if line_length == 0:
                        str_pos = str_pos_eol
                        para_v_spacing = hta_utils.line_space_calc(
                            self.para_size[str_pos], self.para_height[str_pos])
                        v = v + para_v_spacing # Add paragraph vertical spacing
                    else:
                        w_temp = 0
                        stripped_width = 0
                        width_this_line = 0
                        prev_stripped_width = 0
                        strip_group = inkex.etree.Element('g')
                        strip_group.set('id', self.id_new())

                        while i < line_length:
                            word_start = i # Value of i at beginning of the current word.
                            word_group = inkex.etree.Element('g')
                            word_group.set('id', self.id_new())

                            while i < line_length: # Until end of word (or line)
                                character = text_line[i] # character is unicode
                                str_pos = str_pos_eol + i

                                if text_handling == "flowed_svg1":
                                    if self.undefined_height[str_pos]:
                                        no_flowpara_spacing = True

                                char_data = dict()
                                char_data['stroke_scale'] = scale_r
                                char_data['char'] = character
                                char_data['font_size'] = self.text_heights[str_pos]
                                char_data['font_family'] = self.text_families[str_pos]
                                char_data['offset'] = w_temp
                                char_data['vertoffset'] = 0
                                text_align = self.text_aligns[str_pos]

                                # Check to see if this is the last character in the word:
                                last_char = False # Detect last character in "word":
                                if (character.isspace() or character in ["}", ")", "-", "]",
                                    "\u200b"]) and not character == nbsp: # Re-eval for RTL
                                    last_char = True
                                i += 1
                                if i < line_length: # Avoid index out of range
                                    if text_line[i] in ["{", "(", "["]\
                                        and not character == nbsp: # Re-eval for RTL languages
                                        last_char = True
                                w_temp, empty = self.draw_svg_text(char_data, word_group)
                                if not empty: # if most recent glyph is not whitespace:
                                    stripped_width = w_temp
                                if last_char:
                                    break

                            render_strip = False
                            if (stripped_width + self.line_offset) > 0.995 * rect_width:
                                if word_start == line_start: # Single word overflows the box
                                    render_strip = True
                                    width_this_line = stripped_width
                                    strip_group.append( word_group )
                                else: # Render the line only UP UNTIL this word.
                                    render_strip = True
                                    i = word_start
                                    width_this_line = prev_stripped_width
                            elif i >= line_length: # Always render at end of text_line
                                width_this_line = stripped_width
                                strip_group.append( word_group )
                                render_strip = True

                            if not render_strip: # Only add word to the strip's group:
                                strip_group.append( word_group )
                                prev_stripped_width = stripped_width
                            else: # Render the strip of text:
                                self.new_line = True

                                if width_this_line > max_strip_width:
                                    max_strip_width = width_this_line

                                line_max_v_space = 0
                                max_char_height = 0
                                max_para_size = 0
                                no_flowpara_spacing = False

                                j = line_start
                                if text_handling == "flowed_svg1":
                                    if self.undefined_height[str_pos]:
                                        no_flowpara_spacing = True
                                while j < i: # Calculate max height for the strip:
                                    str_pos = str_pos_eol + j
                                    char_height = float(self.text_heights[str_pos])
                                    para_size = float(self.para_size[str_pos])
                                    if char_height > max_char_height:
                                        max_char_height = char_height
                                    if para_size > max_para_size:
                                        max_para_size = para_size
                                    if no_flowpara_spacing:
                                        v_space = hta_utils.line_space_calc(
                                            self.para_size[str_pos],
                                            self.para_height[str_pos])
                                    else:
                                        v_space = hta_utils.line_space_calc(char_height,
                                            self.text_space[str_pos])
                                    if v_space > line_max_v_space:
                                        line_max_v_space = v_space
                                    j = j + 1
                                y_offset =  line_max_v_space / 3  # Heuristic

                                if no_flowpara_spacing:
                                    # Heuristic correction to mimic Inkscape line spacing,
                                    # when flowpara does not specify line height:
                                    line_max_v_space += abs(self.para_size[str_pos] -
                                        max_char_height)/3.5

                                    # Heuristic correction to mimic Inkscape Y offsets:
                                    # when flowpara does not specify line height:
                                    y_offset = 0.5 * line_max_v_space \
                                        - max_char_height/8 - max_para_size/7

                                v = v + line_max_v_space

                                if first_line: # Overall paragraph Y position
                                    first_line = False
                                    y_offs_overall = y_offset
                                    if text_align == "center": # Exactly center first line.
                                        self.line_offset = self.half_max_indent
                                        self.indent_offset = 0

                                if text_align == "center": # Alignment for the strip
                                    x_shift = (rect_width - width_this_line) / 2
                                    x_shift += self.line_offset - self.half_max_indent
                                elif text_align == "end":
                                    x_shift = rect_width - width_this_line - self.line_offset
                                else: # Default: Left aligned
                                    x_shift = self.line_offset

                                strip_list.append([deepcopy(strip_group), x_shift, v])
                                strip_group = inkex.etree.Element('g') # Create next element
                                strip_group.set('id', self.id_new())

                                w_temp = 0
                                stripped_width = 0
                                prev_stripped_width = 0
                                width_this_line = 0
                                line_start = i

                    str_pos_eol = str_pos_eol + extd_line_length

                if self.options.enable_defects:
                    strip_list = hta_utils.vary_line_spacing(strip_list,
                        self.options.leading_var, self.options.rand_seed)

                for strip in strip_list: # Add strips to text block
                    strip_object, x_pos, y_pos = strip
                    strip_object.set('transform', f'translate({x_pos:.7f},{y_pos:.7f})')
                    text_block.append( strip_object )

                t = 'translate('+str(start_x)+','+str(float(start_y) - y_offs_overall)+')'

            else:    # If this is a text object, rather than an SVG 1.2 flowroot object:
                # Case C: text node, using original layout (no reflowing)

                text_align = node_style_dict['text-anchor']

                '''
                CASE C: Handle regular (non-flowroot) text nodes

                Recursively loop through content of the text object, looping through text,
                tspan, and other things as necessary. (A recursive search since style elements
                may be nested.)

                Create multiple lists: One of text content, others of the
                style that should be applied to that content.

                For each line, want to record the plain text, font size per character, text
                alignment, and x,y start values for that line)

                (We may need to eventually handle additional text types and tags, as when
                importing from other SVG sources. We should try to eventually support
                additional formulations of x, y, dx, dy, etc.
                https://www.w3.org/TR/SVG/text.html#TSpanElement )

                Then, loop through those lists, one line at a time, rendering text onto lines.
                If the x or y values changed, assume we've started a new line.

                Note: A text element creates a single line of text; it does not create
                multiline text by including line returns within the text itself. Multiple
                lines of text are created with multiple text or tspan elements.
                '''

                node_info = {'x_pos': start_x, 'y_pos': start_y}
                # Keep track of line number for cases where daughter tspan elements do not
                # have Y positions given. Reset to zero on each text element.
                self.line_number = 0
                self.y_cr_offset = 0.0 # Keep track of "manual" carriage returns in text

                self.recursive_parse_textnode(node, node_info, node_style_dict, 0)

                if self.text_string == "" or self.text_string.isspace():
                    if not self.options.preserve_text:
                        self.nodes_to_delete.append(node)
                    continue # No convertible text in this SVG element.

                letter_vals = list(self.text_string)
                str_len = len(letter_vals)
                strip_group = inkex.etree.Element('g')
                strip_group.set('id', self.id_new())

                w = 0 # Initial spacing offset
                i = 0
                while i < str_len:    # Loop through the entire text of the string.
                    x_startline = float(self.text_x[i])  # We are starting a new line here.
                    y_startline = float(self.text_y[i])

                    while i < str_len:
                        q = letter_vals[i]
                        charfont_height = self.text_heights[i]

                        char_data = dict()
                        char_data['char'] = q
                        char_data['font_family'] = self.text_families[i]
                        char_data['font_size'] = charfont_height
                        char_data['offset'] = w
                        char_data['vertoffset'] = 0
                        char_data['stroke_scale'] = scale_r

                        w, _empty = self.draw_svg_text(char_data, strip_group) # Regular text
                        width_this_line = w
                        w_temp = w

                        # Set the alignment if (A) this is the last character in the string
                        # or if the next piece of the string is at a different position
                        set_alignment = False
                        i_next = i + 1
                        if i_next >= str_len: # End of the string; this is the last character.
                            set_alignment = True
                        elif ((float(self.text_x[i_next]) != x_startline) or
                            (float(self.text_y[i_next]) != y_startline)):
                            set_alignment = True
                        if set_alignment:
                            text_align = self.text_aligns[i]
                            # Not currently supporting text alignment that changes in span;
                            # Uses the text alignment as of the last character.

                            if DEBUG:
                                inkex.debug("text_align: " + text_align)

                            x_shift = 0
                            if text_align == "middle": # when using text-anchor
                                if first_line: # Exactly center first line only
                                    first_line = False
                                    self.line_offset = self.half_max_indent
                                    self.indent_offset = 0
                                strip_center = width_this_line / 2
                                strip_center -= self.line_offset - self.half_max_indent
                                x_shift = x_startline - strip_center

                            elif text_align == "end": # Right-aligned text
                                x_shift = x_startline - width_this_line - self.line_offset
                            else: # Default: Left aligned text
                                x_shift = x_startline + self.line_offset
                            strip_list.append([deepcopy(strip_group), x_shift, y_startline])
                            strip_group = inkex.etree.Element('g') # Create next element
                            strip_group.set('id', self.id_new())

                            self.new_line = True # Used for managing indent defects
                            w = 0
                            i += 1
                            break
                        i += 1    # Only executed when set_alignment is false.

                if self.options.enable_defects:
                    strip_list = hta_utils.vary_line_spacing(strip_list,
                        self.options.leading_var, self.options.rand_seed)
                for strip in strip_list: # Add strips to text block
                    strip_object, x_pos, y_pos = strip
                    strip_object.set('transform', f'translate({x_pos:.7f},{y_pos:.7f})')
                    text_block.append( strip_object )
                t = ""

            if len(strip_group) == 0: # Remove empty strip group shell, if necessary
                lineparent = strip_group.getparent()
                if lineparent is not None:
                    lineparent.remove(strip_group)
            #End cases A & B. Apply transform to text/flowroot object:

            # Simple transformation concatenation may end up like:
            # "translate(-32.477,-204.401)translate(58.688,293.763)scale(0.7758620)"
            # A better approach is to fully compose the transformation, giving
            # a more compact file, at the cost of slightly more processing time:
            if transform is not None:
                t2 = simpletransform.parseTransform(t)
                result = simpletransform.composeTransform(transform2, t2)
                t4 = simpletransform.formatTransform(result)
            else:
                t4 = t
            text_block.set('transform', str(t4))

            if not self.output_generated:
                parent.remove(text_block)    #remove empty group

            #Preserve original element?
            if not self.options.preserve_text and self.output_generated:
                self.nodes_to_delete.append(node)


    def effect(self):
        ''' Main entry point of Hershey Advanced'''
        self.start_time = time.time()

        # Input sanitization:
        self.options.mode = self.options.mode.strip("\"")
        if self.options.font_option is not None:
            self.options.font_option = self.options.font_option.strip("\"")
        self.options.font_face = self.options.font_face.strip("\"")
        self.options.sample_text = self.options.sample_text.strip("\"")

        # Case insensitivity for font names:
        if self.options.font_option is not None:
            self.options.font_option = self.options.font_option.lower()

        # Maximum allowed variations before steering back:
        # Set by the constants here, and limited by the option values (0 - 100 %).
        self.max_baseline_var = self.options.baseline_var * 1.0 / 100.0
        self.max_indent_var = self.options.indent_var * 5.0 / 100.0
        self.max_kern_var = self.options.kern_var * 0.5 / 100.0
        self.max_size_var = self.options.size_var * 1.0 / 100.0
        self.half_max_indent = 0 # Used in horizontal centering of text

        font_option_temp = self.options.font_option # Back up this value
        self.doc_transform = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        self.font_load_fail = False

        self.find_font_files()
        self.font_dict = dict() # Font dictionary - Dictionary of loaded fonts

        if self.options.rand_seed == 1:
            self.s_rand = random.Random(None) # Default value (1); use default random seed method.
        else:
            self.s_rand = random.Random(self.options.rand_seed)  # Use chosen value of random seed.

        if DEBUG:
            inkex.errormsg("rand_seed: "  + str(self.options.rand_seed) + '.')

        self.nodes_to_delete = [] # List of font elements to remove
        # Must save a list and add back at the end, so that any use elements
        # still have the original nodes to reference.

        self.output_generated = False

        self.warn_unflow = False
        self.warn_unkern = False
        self.warn_text_path = False

        self.handle_viewbox()

        self.doc_id_list = {}
        self.id_search(self.document.getroot())

        # Calculate "ideal" effective width of rendered strokes:
        _rendered_stroke_scale = 1 / (96.0 * 800.0) # 1/800 of smaller of page width or height

        self.api_calls = 0
        self.substitution_performed = False
        self.render_width = 1
        if self.svg_width is not None:
            if self.svg_width < self.svg_height:
                self.render_width = self.svg_width * _rendered_stroke_scale
            else:
                self.render_width = self.svg_height * _rendered_stroke_scale

        if self.options.mode == "utilities": # GUI only pseudo-mode
            self.options.mode = self.options.util_mode

        if self.options.mode == "help":
            inkex.errormsg(hta_utils.HELP_TEXT)
        elif self.options.mode in ("glyph_table", "font_table", "list_fonts"):
            preserve_temp = self.options.preserve_text # Temporarily disable preserve_text
            defects_temp = self.options.enable_defects # Temporarily disable enable_defects
            self.options.preserve_text = False
            self.options.enable_defects = False
            self.getposinlayer()
            if self.options.mode == "glyph_table":
                self.glyph_table()
            else:
                self.font_table()
            self.options.preserve_text = preserve_temp
            self.options.enable_defects = defects_temp
        else:   # Process document & perform stroke font substitution
            if self.options.ids:
                # Traverse selected objects
                for one_id in self.options.ids:
                    transform = hta_utils.get_enclosing_transform(
                        self.selected[one_id], self.doc_transform)
                    self.text_substitution([self.selected[one_id]])
                    self.recursively_traverse_svg([self.selected[one_id]],
                        transform, 'visible')
                    self.text_substitution([self.selected[one_id]], True) # Replace original text
            else: # Traverse entire document
                self.text_substitution(self.document.getroot())
                self.recursively_traverse_svg(self.document.getroot(),
                    self.doc_transform, 'visible')
                self.text_substitution(self.document.getroot(), True) # Replace original text

        for element_to_remove in self.nodes_to_delete:
            if element_to_remove is not None:
                parent = element_to_remove.getparent()
                if parent is not None:
                    parent.remove(element_to_remove)

        self.options.font_option = font_option_temp # Restore initial value

        if self.font_load_fail:
            inkex.errormsg('Warning: unable to load SVG stroke fonts.')
            self.document = deepcopy(self.original_document) # Hard revert

        if self.warn_unflow:
            inkex.errormsg('Warning: unable to convert text flowed into a frame.\n'
                           + 'Please use Text > Unflow to convert it prior to use.\n'
                           + 'If you are unable to identify the object in question, '
                           + 'please contact technical support for help.')
            self.document = deepcopy(self.original_document) # Hard revert

        if self.warn_unkern:
            inkex.errormsg('Warning: unable to render text.\n'
                + 'Please use Text > Remove Manual Kerns to convert it prior to use .')
            self.document = deepcopy(self.original_document) # Hard revert

        if self.warn_text_path:
            inkex.errormsg('Warning: unable to convert text on path.\n'
                           + 'Please use Text > Remove from path.\n'
                           + 'If you are unable to identify the object in question, '
                           + 'please contact technical support for help.')
            self.document = deepcopy(self.original_document) # Hard revert

        elapsed_time = time.time() - self.start_time

        if DEBUG:
            inkex.errormsg("Hershey Rendering took: {0:03f} seconds".format(elapsed_time))
            inkex.errormsg("API Calls: " + str(self.api_calls))


if __name__ == '__main__':
    e = HersheyAdv()
    exit_status.run(e.affect)
