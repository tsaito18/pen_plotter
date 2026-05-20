'''
axidraw_merge_conf.py
Part of Hershey Advanced


Version 5.1.0, dated 2022-07-22

Copyright 2022 Windell H. Oskay, All rights reserved
Evil Mad Scientist Laboratories, www.evilmadscientist.com


Configuration file for AxiDraw Merge

We encourage you to freely tune these values as needed to match the
 behavior and performance of your AxiDraw to your application and taste.

If you are operating the AxiDraw from within Inkscape (either within the
application from the Extensions menu or from the command line), please
set your preferences within Inkscape, using the AxiDraw Control dialog.
(The values listed here are ignored when called via Inkscape.)

If you are operating the AxiDraw in "standalone" mode, that is, outside
of the Inkscape context, then please set your preferences here or via
command-line arguments. (Preferences set within Inkscape -- via the
AxiDraw Control dialog -- are ignored when called via the command line.)
Recommended practice is to adjust and test settings from within the
Inkscape GUI, before moving to stand-alone (CLI-based) control.

'''

# Data merge parameters
first_row = 1            # Merge Start Row.
last_row = 0             # Merge End row. Value of 0: Continue until end of data
page_delay = 10          # Seconds to delay between copies
single_row = 1           # Specified Row to Merge

# Plotting parameter defaults:

speed_pendown = 2000      # Maximum plotting speed, when pen is down (1-100)
speed_penup = 8000        # Maximum transit speed, when pen is up (1-100)
accel = 75              # Acceleration rate factor (1-100)

pen_pos_up = 0.5         # Height of pen when raised (0-100)
pen_pos_down = 5       # Height of pen when lowered (0-100)
laser_power = 50
laser_focus_power =3

pen_rate_raise = 5000     # Rate of raising pen (1-100)
pen_rate_lower = 5000     # Rate of lowering pen (1-100)

pen_delay_up = 0        # Optional delay after pen is raised (ms)
pen_delay_down = 0      # Optional delay after pen is lowered (ms)

const_speed = False     # Use constant velocity mode when pen is down.
report_time = False     # Report time elapsed.
default_layer = 1       # Layer(s) selected for layers mode (1-1000).

page_delay = 15         # Optional delay between copies (s).
plot_file = False

preview = False         # Preview mode; simulate plotting only. True or False.

model = 1               # AxiDraw Model (1-6).
                            # 1: AxiDraw V2 or V3 (Default).
                            # 2: AxiDraw V3/A3 or SE/A3.
                            # 3: AxiDraw V3 XLX.
                            # 4: AxiDraw MiniKit.
                            # 5: AxiDraw SE/A1.
                            # 6: AxiDraw SE/A2.

penlift = 1             # pen lift servo configuration (1-3). 
                            # 1: Default for AxiDraw model
                            # 2: Standard servo (lowest connector position)
                            # 3: Narrow-band brushless servo (3rd position up)

port = None             # Serial port or named AxiDraw to use.
                            # None (Default) will plot to first unit located.

port_config = 0         # Serial port behavior option (0-2)
                            # 0: Plot to first unit found, unless port is specified (Default),
                            # 1: Plot to first AxiDraw unit located
                            # 2: Plot to a specific AxiDraw only, given by port.

auto_rotate = True     # Auto-select portrait vs landscape orientation

resolution = 1          # Resolution: (1-2):
                            # 1: High resolution (smoother, slightly slower) (Default)
                            # 2: Low resolution (coarser, slightly faster)

rendering = 3           # Preview mode rendering option (0-3)
                            # 0: Do not render previews
                            # 1: Render only pen-down movement
                            # 2: Render only pen-up movement
                            # 3: Render all movement

reordering = 0          # Plot optimization option (0-4; 3 is deprecated)
                            # 0: Least; Only connect adjoining paths (Default)
                            # 1: Basic; Also reorder paths for speed
                            # 2: Full; Also allow path reversal
                            # 4: None; Strictly preserve file order

random_start = False    # Randomize start locations of closed paths. (Default: False)

hiding = False          # Hidden-line removal. Default: False

webhook = False         # Enable webhook alerts. Default: False

webhook_url = None      # URL for webhook alerts

# The resolution parameter selects the microstepping level used on the motors. Effective motor
# resolution is approx. 1437 or 2874 steps per inch, along the native motor axes of the AxiDraw,
# (X+Y) and (X-Y), not along the XY axes of the machine.

# Text substitution parameters

font_option = "EMSBird" # Default font face

letter_spacing = 100   # Override letter spacing (percent).    Range: 50 - 400
word_spacing = 100     # Override word spacing (percent).      Range: 50 - 600

enable_defects = False # Enable Handwriting Defects. True or False.

leading_var = 15       # Variation in line spacing, when enable_defects is True (%). (0-100).
baseline_var = 15      # Variation in text baseline, when enable_defects is True (%). (0-100).
indent_var = 15        # Variation in indent, when enable_defects is True (%).        (0-100).
kern_var = 15          # Variation in kerning, when enable_defects is True (%).       (0-100).
size_var = 15          # Variation in font size, when enable_defects is True (%).     (0-100).


'''
Additional user-adjustable control parameters:

These parameters are adjustable only from this config file, and are not visible
from within the Inkscape GUI.
'''

options_message = True  # If True (default), display an advisory message if Apply is clicked
                        #   in the AxiDraw Merge GUI, while in tabs that have no effect.
                        #   (Clicking Apply on these tabs has no effect other than the message.)
                        #   This message can prevent the situation where one clicks Apply on an
                        #   Options tab and then waits a few minutes before realizing that
                        #   no plot has been initiated.

digest = 0              # Plot digest output option. Reserved for future use; do not enable.
