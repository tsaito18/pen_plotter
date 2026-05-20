'''
hershey_conf.py
Part of Hershey Advanced


Version 5.4.0, dated 2023-01-10

Copyright 2023 Windell H. Oskay, All rights reserved
Evil Mad Scientist Laboratories, www.evilmadscientist.com


Configuration file for Hershey Advanced

If you are operating Hershey Advanced from within Inkscape
 (either within the application from the Extensions menu or from the
 command line), please set your preferences within Inkscape, using
 the Hershey Text Advanced dialog under Extensions > Text.
 (The values listed here are ignored when called via Inkscape.)

If you are operating the AxiDraw in "standalone" mode, that is, outside
 of the Inkscape context, then please set your preferences here or via
 command-line arguments. (Preferences set within Inkscape -- via the
 AxiDraw Control dialog -- are ignored when called via the command line.)
 We suggest adjusting and testing settings from within the Inkscape
 GUI, before moving to stand-alone control.
'''

mode = "render"         # General mode of operation: render, glyph_table, or font_table
                        # Default: render

preserve_text = False   # Preserve original text. True or False.

letter_spacing = 100    # Override letter spacing (%).    Range: 50 - 400
word_spacing = 100      # Override word spacing (%).      Range: 50 - 600

reflow_svg2 = True      # Re-flow SVG2-style flowed text when possible

flow_fit = True         # Resize blocks of flowed text, down only, to fit the width of the
                        #   text area, when text variations would otherwise make it overflow.

enable_defects = False  # Enable Handwriting Defects. True or False.

# Allowed range on the following five variables is 0 - 100
leading_var = 15        # Variation in line spacing, when enable_defects is True (%)
baseline_var = 15       # Variation in text baseline, when enable_defects is True (%)
indent_var = 15         # Variation in indent, when enable_defects is True (%)
kern_var = 15           # Variation in kerning, when enable_defects is True (%)
size_var = 15           # Variation in font size, when enable_defects is True (%)

rand_seed = 1           # Random Seed. If default value (1) is left in place, execution time will
                        #   be used as the random seed. However, you can override this and provide
                        #   your own seed when it is necessary to use the same seed repeatedly.

font_option = "other"     # Defaults to "other", so that you can use font_face directly
font_face = "HersheySans1"  # Default font face

sample_text = "The five boxing wizards jump quickly"  # Sample text for font table


# Scriptalizer support -- for use with Quantum Enterprises fonts that support Scriptalizer use

script_enable = True    # Enable Scriptalizer use for use with compatible Quantum Enterprises fonts.
                        #   Use requires an active internet connection and applicable fonts.
                        #   Set to False to disable this feature. Visit
                        #   http://quantumenterprises.co.uk/slf to learn more about these fonts.

script_timeout = 5      # Timeout, seconds, for communicating with Scriptalizer server. Default: 5.0

script_mistakes = 0     # Rate of adding "mistakes" in the Scriptalizer. Set to 0 to disable.
                        # A value of 100 would indicate 1 error per 100 words on average

script_quiet = True     # If true, mute (quietly ignore) any errors encountered when accessing the
                        #    Scriptalizer.  Set to False to report any errors encountered.
