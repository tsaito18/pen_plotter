'''
hta_utils.py
Part of Hershey Advanced

Version 5.4.0, dated 2023-01-10

Copyright 2023 Windell H. Oskay, All rights reserved
Evil Mad Scientist Laboratories, www.evilmadscientist.com

This file contains some utility functions used within Hershey Advanced.
'''

import random
from .plot_utils_import import from_dependency_import
plot_utils = from_dependency_import('plotink.plot_utils') # https://github.com/evil-mad/plotink
simpletransform = from_dependency_import('ink_extensions.simpletransform')
inkex = from_dependency_import('ink_extensions.inkex')

# pylint: disable=pointless-string-statement

DEBUG = False

HELP_TEXT = '''====== Hershey Advanced Help ======

The Hershey Advanced extension is designed to replace text in your document (either
selected text or all text) with specialized "stroke" or "engraving" fonts
designed for plotters.

Whereas regular "outline" fonts (e.g., TrueType) work by filling in the region
inside an invisible outline, stroke fonts are composed only of individual lines
or strokes with finite width; much like human handwriting when using a physical
pen.

Stroke fonts are most often used for creating text-like paths that computer
controlled drawing and cutting machines (from pen plotters to CNC routers) can
efficiently follow.

For a general introduction to stroke fonts, please visit:
  www.evilmadscientist.com/go/hershey


  ==== Basic operation ====

To use Hershey Advanced, start with a document that contains text objects.
Select the "Render" tab of Hershey Advanced, and choose a font face from the
pop-up menu.

When you click Apply, it will render all text elements on your page with the
selected stroke-based typeface. If you would like to convert only certain text
elements, click Apply with just those elements selected.

If the "Preserve original text" box is checked, then the original text elements
on the page will be preserved even when you click Apply. If it is unchecked,
then the original font elements will be removed once rendered.

Various handwriting-like defects can be enabled through the "Options" tab.

You can generate a list of available SVG fonts or a list of all glyphs available
in a given font by using the tools available on the "Utilities" tab.


   ==== How Hershey Advanced works ====

Hershey Advanced works by performing font substitution, starting with the text
in your document and replacing it with paths generated from the characters in
the selected SVG font.

Hershey Advanced uses fonts in the SVG font format. While SVG fonts are one of
the few types that support stroke-based characters, it is important to note that
converting an outline font to SVG format does not convert it to a stroke based
font. Indeed, most SVG fonts are actually outline fonts.

This extension *does not* convert outline fonts into stroke fonts, nor does it
convert other fonts into SVG format. Its sole function is to replace the text
in your document with paths from the selected SVG font.


   ==== Using an external SVG font ====

To use an external SVG font -- one not included with the distribution -- select
"Other" for the name of the font in the pop-up menu on the "Render" tab. Then,
do one of the following:

(1) Add your SVG font file (perhaps "example.svg") to the "svg_fonts" directory
within your Inkscape extensions directory, and enter the name of the font
("example") in the "Other SVG font name or path" box on the "Render" tab.

or

(2) Place your SVG font file anywhere on your computer, and click the "..."
button next to select the font.


   ==== Using SVG fonts: Advanced methods ====

In addition to using a single SVG font for substitution, you can also use
font name mapping to automatically use particular stroke fonts in place of
specific font faces, to support various automated workflows and to support
the rapid use of multiple stroke font faces within the same document.


Several SVG fonts are included with this distribution, including both
single-stroke and multi-stroke fonts. These fonts are included within the
"svg_fonts" directory within your Inkscape extensions directory.

You can select the font that you would like to use from the pop-up menu on the
"Render" Tab. You can also make use of your own SVG fonts.

Order of preference for SVG fonts:

(1) If there is an SVG font with name matching that of the font for a given
piece of text, that font will be used. For example, if the original text is in
font "FancyScript" and there is a file in svg_fonts with name FancyScript.svg,
then FancyScript.svg will be used to render the text.

(2) Otherwise (if there is no SVG font available matching the name of the font
for a given block of text), the face selected from the "Font face" pop-up menu
will be used as the default font when rendering text with Hershey Advanced.

(3) You can also enter text in the "Other SVG Font" box, which can represent one
of the following: (i) a font name (for a font located in the svg_fonts directory),
(ii) the path to a font file elsewhere on your computer, or (iii) the path to a
directory containing (one or more) font files.

(3a) Using a font name:
If you move a custom SVG font file into your svg_fonts directory, then you can
enter the name of the SVG font in the "Other SVG Font" text box and select
"Other" from the pop-up menu. Then, the named font will be used as the default.

(3b) Using a file path:
If you select or enter the path to an SVG font file in the "Other SVG Font" box
and select "Other" from the pop-up menu. Then, that font will be used as the
default. All SVG fonts located in the same directory as that font file will
also be available for name-based font substitution. If there are multiple
font-name matches, files in an external directory take precedence over ones in
the svg_fonts directory.

(3c) Using a directory path:
If you enter the path to a directory containing SVG font files in the
"Other SVG Font" text box, then all SVG font files files in that directory
will be available for name-based font substitution. If there are multiple
font-name matches, files in an external directory take precedence over ones
in the svg_fonts directory.



Tips about using these methods with your own custom fonts:

(A) These methods can be used to render different text elements with different
SVG font faces. You can even rename a font -- either your own custom one or one
of the bundled ones -- to match the name of a font that you're using. For
example, if you rename a script font you name a font to "Helvetica.svg",
then all text in Helvetica will be replaced with that SVG font.

(B) Using a directory path (3c) is a particularly helpful method if you do
not have access to modify the svg_fonts directory.


   ==== Scriptalizer font support ====

Certain commercial stroke fonts from Quantum Enterprises support a feature
called "Scriptalizer" that performs automatic character substitution so that the
output text can use more than one glyph, in different places, to represent the
same character. This can improve the "realism" in handwriting-like applications.

If you are using Scriptalizer-compatible SVG fonts from Quantum Enterprises,
text in those fonts will *automatically* be fed and processed through the
Scriptalizer web service, immediately before rendering into paths.

Some things to note about this feature:

(1) This feature is only used when the SVG font that will be used to render text
is from Quantum Enterprises. It is not active in any other circumstance.

(2) The text that you are converting will be sent in a standard encrypted web
request (an HTTPS POST request) for processing by the Scriptalizer web server.
This feature requires that you have an active and working internet connection.

(3) Scriptalizer error messages are normally muted. If (for example) there is no
internet connection, or if your font is unrecognized by the Scriptalizer web
server, no error message will be produced. The Hershey Advanced software will
work normally, with the exception that no Scriptalizer substitution is
performed. If you wish it to report errors, you can open the configuration file,
hershey_conf.py, and change the "script_quiet" option from True to False.

(4) Scriptalizer support can be disabled by opening up the configuration file,
hershey_conf.py, and changing the "script_enable" option from True to False.

(5) The Scriptalizer can optionally add "mistakes" to the text at some given
rate. That rate is set to zero (no mistakes) by the "script_mistakes" option in
hershey_conf.py, and can be edited if you would like to enable mistakes.


   ==== Limitations ====

This extension renders text into non-editable paths, generated from the
character geometry of SVG fonts. Once you have rendered the text, the resulting
paths can be edited with path editing tools, but not text editing tools.

Since this extension works by a process of font substitution, text spanning a
single line will generally stay that way, whereas text flowed in a box (that
may span multiple lines) will be re-flowed from scratch. Style information such
as text size and line spacing can be lost in some cases.

We recommend that you use the live preview option to achieve best results with
this extension.


Copyright 2022 Windell H. Oskay, All rights reserved
Evil Mad Scientist Laboratories, www.evilmadscientist.com'''


def strip_quotes(fontname):
    '''
    A multi-word font name may have a leading and trailing
    single or double quotes, depending on the source. Remove them.
    '''

    if fontname.startswith("'") and fontname.endswith("'"):
        return fontname[1:-1]
    if fontname.startswith('"') and fontname.endswith('"'):
        return fontname[1:-1]
    return fontname


def line_space_calc(local_char_height, line_height_string):
    ''' Calculate line spacings '''

    try: # detect case of number without units:
        line_height_float = float(line_height_string)
        return local_char_height * line_height_float # Relative spacing, numeric
    except ValueError:
        _value, unit = plot_utils.parseLengthWithUnits(line_height_string)
        scaled_val = plot_utils.unitsToUserUnits(line_height_string)

    if unit == "%":
        return local_char_height * scaled_val # Relative spacing, percentage
    return scaled_val # Absolute size for line height


def inherit_styles(parent_styles, node_style_parsed, interparagraph = True):
    '''
    Parse style dict of inner node and supersede
    parent definitions when local styles are defined
    '''

    default_styles = dict()
    default_styles['font-family'] = 'sans-serif'
    default_styles['text-anchor'] = 'left'
    default_styles['text-align'] = 'left'
    default_styles['font-size'] = plot_utils.unitsToUserUnits("16px") # Default font size
    default_styles['line-height'] = "1.25" # Inkscape default 1.25
    default_styles['letter-spacing'] = 0.0
    default_styles['word-spacing'] = 0.0

    # Separate styles for spacing between paragraphs:
    default_styles['font-size-paragraph'] = plot_utils.unitsToUserUnits("12px")
    default_styles['line-height-paragraph'] = "1.25" # Inkscape default paragraph spacing

    if parent_styles is None:
        parent_styles = default_styles  # Use defaults if there is no parent

    if node_style_parsed is None: # No new style information provided
        return parent_styles

    # Copy, not assignment, so that each represents an independent dict:
    new_styles = parent_styles.copy()

    # Style attributes set via a CSS length string & stored as a float:
    for attrib in ['font-size', 'letter-spacing', 'word-spacing', ]:
        value = node_style_parsed.get(attrib) # Defaults to None, preventing KeyError
        if value:
            value = value.lower()
            if value in ['inherit']:
                new_styles[attrib] = parent_styles[attrib]
            else:
                new_styles[attrib] = plot_utils.unitsToUserUnits(value)

            if attrib == "font-size":
                if interparagraph:
                    new_styles['font-size-paragraph'] = new_styles['font-size']

    # Style attributes set and stored as strings:
    for attrib in ['font-family', 'text-anchor', 'text-align', 'line-height',]:
        value = node_style_parsed.get(attrib) # Defaults to None, preventing KeyError
        if value:
            # Special quote-stripping: Needed for font-family and harmless otherwise:
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            # normal only applies to line height, but should not occur otherwise.
            if value in ['normal']:
                new_styles[attrib] = default_styles[attrib]
            elif value in ['inherit']:
                new_styles[attrib] = parent_styles[attrib]
            else:
                new_styles[attrib] = value

            if attrib == "line-height":
                height_str = new_styles['line-height'].strip()
                if height_str[-2:] == 'em': # Same as font height or "lines"; strip the units:
                    new_styles['line-height'] = float(height_str[:-2])
                if height_str[-2:] == 'ex': # ex units; not fully supported; approximate as 50%
                    new_styles['line-height'] = 0.5 * float(height_str[:-2]) # of font height
                if interparagraph:
                    new_styles['line-height-paragraph'] = new_styles['line-height']

    return new_styles


def get_enclosing_transform(node, default_transform):
    '''Determine cumulative transform which node inherits from its chain of ancestors.'''

    node = node.getparent()
    if node is not None:
        parent_transform = get_enclosing_transform(node, default_transform)
        node_transform = node.get('transform')
        if node_transform is None:
            return parent_transform
        the_transform = simpletransform.parseTransform(node_transform)
        if parent_transform is None:
            return the_transform
        return simpletransform.composeTransform(parent_transform, the_transform)
    return default_transform


def parse_svg_font(node_list, qes_in=None):
    '''
    Parse an input svg etree, searching for an SVG font. If one is found, parse it and return
    a "digest" containing structured information from the font. See below for more about the
    digest format.

    If the font is not found cannot be parsed, return None.

    Notable limitations:

    (1) This function only parses the first font face found within the tree. We may, in the
    future, support multiple fonts within an SVG file.

    (2) We are only processing left-to-right and horizontal text, not vertical text nor RTL.

    (3) This function currently performs only certain recursive searches, within the <defs>
    element. It will not discover fonts nested within groups or other elements. So far as we
    know, that is not a limitation in practice. (If you have a counterexample please contact
    Evil Mad Scientist tech support and let us know!)

    (4) Kerning details are not fully implemented yet. Also, hkern elements can use g1, g2 OR
    u1, u2, but not g1,u2 or u1, g2. (This does not appear to be a limitation in pracice.)
    '''

    digest = None
    qes = False
    if qes_in:
        qes = True
    if node_list is None:
        if DEBUG:
            inkex.errormsg('Empty node list at parse_svg_font')
        return None

    for node in node_list:
        if node.tag == inkex.addNS('metadata', 'svg') or node.tag == 'metadata':
            if node.text is not None:
                if "Quantum Enterprises" in node.text:
                    qes = True
                    if node.get('data-scriptalize') == "standard":
                        qes = False
            if DEBUG:
                inkex.errormsg('Font Scriptalizable: ' + str(qes))

        if node.tag == inkex.addNS('defs', 'svg') or node.tag == 'defs':
            return parse_svg_font(node, qes) # Recursive call

        if node.tag == inkex.addNS('font', 'svg') or node.tag == 'font':
            if DEBUG:
                inkex.errormsg('parse_svg_font: Now parsing font')
            '''
            === Internal structure for storing font information ===

            We parse the SVG font file and create a keyed "digest"
            from it that we can use while rendering text on the page.

            This "digest" will be added to a dictionary that maps
            each font family name to a single digest.

            The digest itself is a dictionary with the following
            keys, some of which may have empty values. This format
            will allow us to add additional keys at a later date,
            to support additional SVG font features.

                font_id (a string)

                font_family (a string)

                glyphs
                    A dictionary mapping unicode points to a specific
                    dictionary for each point.  See below for more about
                    the key format.
                    The dictionary for a given point will include keys:
                        glyph_name (string)
                        horiz_adv_x (numeric)
                        d (string)

                missing_glyph
                    A dictionary for a single code point, with keys:
                        horiz_adv_x (numeric)
                        d (string)

                hkern_u
                    A list of 3-tuples, each containing 3 strings.
                    For kerning specified with unicode points

                hkern_g
                    A list of 3-tuples, each containing 3 strings.
                    For kerning specified with glyph names

                geometry
                    A dictionary containing geometric data
                    Keys will include:
                        horiz_adv_x (numeric) -- Default value
                        units_per_em (numeric)
                        ascent (numeric)
                        descent (numeric)
                        x_height (numeric)
                        cap_height (numeric)
                        bbox  (string)
                        underline_position (numeric)
                scale
                    A numeric scaling factor computed from the
                    units_per_em value, which gives the overall scale

                qes (boolean) - True if Scriptalizer support enabled
            '''

            digest = {}
            geometry = {}
            glyphs = {}
            missing_glyph = {}
            hkern_u = []
            hkern_g = []

            digest['font_id'] = node.get('id')

            horiz_adv_x = node.get('horiz-adv-x')

            if horiz_adv_x is not None:
                geometry['horiz_adv_x'] = float(horiz_adv_x)
            # Note: case of no horiz_adv_x value is not handled.

            glyph_tag = inkex.addNS('glyph', 'svg')
            ff_tag = inkex.addNS('font-face', 'svg')
            mg_tag = inkex.addNS('missing-glyph', 'svg')
            hk_tag = inkex.addNS('hkern', 'svg')

            for element in node:
                if element.tag in ('glyph', glyph_tag):
                    # First, because it is the most common element
                    uni_text = element.get('unicode')
                    if not uni_text:
                        continue # Can't use this point if no unicode mapping.

                    '''
                    About the unicode attribute:

                    While the unicode value for a given glyph may be specified in a variety
                    of ways, we need to be able to retrieve the glyph that corresponds to a
                    particular unicode point.

                    Thus, we parse the unicode attribute and rewrite it in a consistent style
                    that allows us to reliably use it as a key for our glyph dictionary.

                    The unicode attribute may:
                    - Consist of a single "ascii" character: unicode="b"
                    - Contain ligatures sequences, consisting of
                        multiple "ascii" characters, like
                        unicode="ffl" or unicode="fF"
                    - Consist of an entity expressed in hex,
                        like unicode="&#x3c;"
                    - Consist of an entity expressed in decimal,
                        like unicode="&#102;"
                    - Contain multiple hex or decimal entities,
                        like unicode="&#x66;&#x66;&#x6c;"
                        or unicode="&#102;&#102;&#108;"

                    [ Reference: https://www.w3.org/TR/SVG11/fonts.html ]

                    These *should* all be decoded by the initial lxml parsing of the SVG font,
                    which means that we store a unicode string with the decoded values.

                    Aside: This program does not yet have an implementation for displaying
                    alternate glyphs; we need an example  SVG font and SVG file using that
                    font with alternate glyphs in order to implement and test this feature.
                    If you have one, please contact Evil Mad Scientist technical support.
                    '''

                    if uni_text in glyphs: # Skip if unicode point is already in glyph list.
                        continue # Only one glyph per code point is currently supported.

                    glyph_dict = {'glyph_name': element.get('glyph-name')}

                    horiz_adv_x = element.get('horiz-adv-x')

                    if horiz_adv_x is not None:
                        glyph_dict['horiz_adv_x'] = float(horiz_adv_x)
                    else:
                        glyph_dict['horiz_adv_x'] = geometry['horiz_adv_x']

                    glyph_dict['d'] = element.get('d') # SVG path data
                    glyphs[uni_text] = glyph_dict

                elif element.tag in ('font-face', ff_tag):
                    digest['font_family'] = element.get('font-family')
                    units_per_em = element.get('units-per-em')

                    if units_per_em is None: # Default: 1000, per SVG specification.
                        geometry['units_per_em'] = 1000.0
                    else:
                        geometry['units_per_em'] = float(units_per_em)

                    ascent = element.get('ascent')
                    if ascent is not None:
                        geometry['ascent'] = float(ascent)

                    descent = element.get('descent')
                    if descent is not None:
                        geometry['descent'] = float(descent)

                    '''
                    # Skip these attributes that we are not currently using
                    geometry['x_height'] = element.get('x-height')
                    geometry['cap_height'] = element.get('cap-height')
                    geometry['bbox'] = element.get('bbox')
                    geometry['underline_position'] = element.get('underline-position')
                    '''

                elif element.tag in ('missing-glyph', mg_tag):
                    horiz_adv_x = element.get('horiz-adv-x')

                    if horiz_adv_x is not None:
                        missing_glyph['horiz_adv_x'] = float(horiz_adv_x)
                    else:
                        missing_glyph['horiz_adv_x'] = geometry['horiz_adv_x']

                    missing_glyph['d'] = element.get('d') # SVG path data
                    digest['missing_glyph'] = missing_glyph

                elif element.tag in ('hkern', hk_tag):

                    g_1 = element.get('g1')
                    g_2 = element.get('g2')
                    g_k = element.get('k')

                    if None not in (g_1, g_2, g_k):
                        hkern_g.append((g_1, g_2, g_k))
                        continue
                    u_1 = element.get('u1')
                    u_2 = element.get('u2')
                    if None not in (u_1, u_2, g_k):
                        hkern_u.append((u_1, u_2, g_k))

            # Main scaling factor
            digest['scale'] = 1.0 /  geometry['units_per_em']

            digest['glyphs'] = glyphs
            digest['geometry'] = geometry
            digest['hkern_g'] = hkern_g
            digest['hkern_u'] = hkern_u
            digest['qes'] = qes
            qes = False
            return digest
    return None


def vary_line_spacing(strip_list, amount, rand_seed=1):
    '''
    Given a list of "text strip" objects, vary their Y spacing by a random-walk scaled amount.
    Maximum deviation at 100% amount is hard-coded to be 50% (+/- 25%) of default line spacing.
    Inputs:
        * strip_list, a list of 3-tuples of (etree element, x position, y position)
            * The y position values are assumed to monotonically increase.
        * amount, a percentage value (0-100) of how much variation to apply
        * rand_seed (optional). If 1, use default random seed method.
    Returns: Modified strip_list, with y positions varied. First and last lines are left
             unchanged, as are lines unevenly spaced for reasons of font size changes. Changes
             are only made to the middle positions of 3 or more uniformly spaced text strips.
    '''

    if len(strip_list) < 3:
        return strip_list
    if amount == 0:
        return strip_list

    original_y = [text_strip[2] for text_strip in strip_list]

    matched_spacing_sets = find_evenly_spaced_sets(original_y)
    if len(matched_spacing_sets) == 0:
        return strip_list

    # Begin with random-walk scale, with unit average step size
    if rand_seed == 1:
        s_rand = random.Random(None) # Default value (1); use default random seed method.
    else:
        s_rand = random.Random(rand_seed)  # Use specified random seed.

    for sub_range in matched_spacing_sets:
        # Add variation to "middle" line spacing, within blocks of at least three
        # evenly spaced lines

        fragment_y = original_y[sub_range[0]:sub_range[1]] # the block

        min_y = fragment_y[0]
        y_pos = 0
        new_y_positions = []
        for _step in fragment_y:
            new_y_positions.append(y_pos) # first element will always be 0.
            y_pos += 1 + (s_rand.random() - 0.5) * (amount / 200) # Divisor sets overall scale.

        scale_factor = (fragment_y[-1] - min_y) / new_y_positions[-1]

        index = sub_range[0] + 1
        index_2 = 1
        while index < sub_range[1]:
            strip_list[index][2] = new_y_positions[index_2] * scale_factor + min_y
            index += 1
            index_2 += 1

    return strip_list


def find_evenly_spaced_sets(number_list):
    '''
    Given a list of numbers, determine which of these numbers are
    evenly spaced (within 1%). Return a list of 2-element lists, indicating
    the first and last index of the evenly spaced numbers. The numbers in the input list
    should monotonically increase.
    Example input: [1.2, 3, 5, 7, 9, 12, 15, 18, 21, 24, 27, 27.5, 29, 30.5]
    Example output: [[1, 4], [4, 10], [11, 13]]
    '''

    matched_spacing_sets = []
    max_index = len(number_list) - 1
    index = 0
    while index <= max_index:
        y_pos = number_list[index]
        first_line_index = index
        last_line_index = index
        index_adv = 1
        if (index + index_adv) > max_index:
            break
        first_spacing = number_list[index + 1] - y_pos
        while True:
            index_adv += 1
            if (index + index_adv) > max_index:
                if (last_line_index - first_line_index) >= 1:
                    matched_spacing_sets.append([first_line_index,last_line_index])
                break
            spacing = number_list[index + index_adv] - number_list[index + index_adv - 1]
            if .99 < (spacing/first_spacing) < 1.01 : # 1% tolerance on spacing equality
                last_line_index = index + index_adv
                continue
            if (last_line_index - first_line_index) >= 1:
                matched_spacing_sets.append([first_line_index,last_line_index])
                index = last_line_index - 1
            break
        index += 1
    return matched_spacing_sets
