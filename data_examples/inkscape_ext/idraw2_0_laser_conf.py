'''
# idraw_conf.py
# Part of the iDraw driver software
'''

# DEFAULT VALUES
mode = 'plot'           # Operational mode or GUI tab. Default: 'plot'

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

const_speed = False     # Use constant velocity mode when pen is down
report_time = False     # Report time elapsed
default_layer = 1       # Layer(s) selected for layers mode (1-1000)
manual_cmd = 'fw_version'   # Manual command to execute when in manual mode.
                            # Default 'fw_version'
dist = 1.0              # Distance to walk in "walking" manual commands or changing
                            # resume position in the APIs. Variable units. Default 1.0
copies = 1              # Copies to plot, or 0 for continuous plotting. Default: 1
page_delay = 15         # Optional delay between copies (s).
plot_file = False

preview = False         # Preview mode; simulate plotting only.
rendering = 3           # Preview mode rendering option (0-3):
                            # 0: Do not render previews
                            # 1: Render only pen-down movement
                            # 2: Render only pen-up movement
                            # 3: Render all movement (Default)

model = 1               # iDraw Model (1-4)
                            # 1: iDraw V2 or V3 (Default)
                            # 2: iDraw V3/A3 or SE/A3
                            # 3: iDraw V3 XLX
                            # 4: iDraw MiniKit
                         
port = None             # Serial port or named iDraw to use
                            # None (Default) will plot to first unit located

port_config = 0         # Serial port behavior option (0-2)
                            # 0: Plot to first unit found, unless port is specified (Default)
                            # 1: Plot to first iDraw unit located
                            # 2: Plot to a specific iDraw only, given by port

auto_rotate = True      # Auto-select portrait vs landscape orientation
                            # Default: True

reordering = 0          # Plot optimization option
                            # 0: Preserve order of objects given in SVG file (Default)
                            # 1: Reorder objects, preserving path orientation
                            # 2: Reorder objects, allow path reversal

random_start = False    # Randomize start locations of closed paths. (Default: False)
hiding = False          # Hidden-line removal. Default: False
webhook = False         # Enable webhook alerts when True
                            # Default False

webhook_url = None      # URL for webhook alerts. Default None

digest = 0              # Plot digest output option. (NOT supported in Inkscape context.)
                            # 0: Disabled; No change to behavior or output (Default)
                            # 1: Output "plob" digest, not full SVG, when saving file
                            # 2: Disable plots and previews; generate digest only

progress = False        # Enable progress bar display in iDraw CLI, when True
                            # Default False
                            # This option has no effect in Inkscape or Python API contexts.

resolution = 1          # Resolution: (1-2):
                            # 1: High resolution (smoother, slightly slower) (Default)
                            # 2: Low resolution (coarser, slightly faster)

# Effective motor resolution is approx. 1437 or 2874 steps per inch, in the two modes respectively.
# Note that these resolutions are defined along the native axes of the machine (X+Y) and (X-Y),
# not along the XY axes of the machine. This parameter chooses 8X or 16X motor microstepping.

'''
Additional user-adjustable control parameters:

Values below this point are configured only in this file, not through the user interface(s).
'''

servo_timeout = 60000   # Time, ms, for servo motor to power down 
                        #   after last movement command  (default: 60000)
                        #   This feature requires EBB v 2.5 hardware (with USB
                        #   micro not USB mini connector) and firmware version
                        #   2.6.0 or newer

check_updates = True    # If True, allow iDraw Control to check online to see
                        #    what the current software version is, when you
                        #    query the version. Set to False to disable. Note that
                        #    this is the only internet-enabled function in the
                        #    iDraw software.

smoothness = 10.0       # Curve smoothing (default: 10.0)

cornering = 10.0        # Cornering speed factor (default: 10.0)

use_pwm_out = 1         # If True, enable digital output pin PWM, which will be high (5V)
                        #   when the pen is down, and low otherwise. Can be used to control
                        #   external devices like valves, relays, or lasers.
                        # 0: only pen out (default)
                        # 1: only pwm out
                        # 2: pwm out and pen out

auto_rotate_ccw = True  # If True (default), auto-rotate is counter-clockwise when active.
                        #   If False, auto-rotate direction is clockwise.

options_message = True  # If True (default), display an advisory message if Apply is clicked
                        #   in the iDraw Control GUI, while in tabs that have no effect.
                        #   (Clicking Apply on these tabs has no effect other than the message.)
                        #   This message can prevent the situation where one clicks Apply on the
                        #   Options tab and then waits a few minutes before realizing that
                        #   no plot has been initiated.

report_lifts = False    # Report number of pen lifts when reporting plot duration (Default: False)

auto_clip_lift = True   # Option applicable to the Interactive Python API only.
                        #   If True (default), keep pen up when motion is clipped by travel bounds.

# Colors used to represent pen-up and pen-down travel in preview mode:
preview_color_up = 'LightPink' # Pen-up travel color. Default: LightPink; rgb(255, 182, 193)
preview_color_down = 'Blue'    # Pen-up travel color. Default: Blue; rgb(0, 0, 255)

skip_voltage_check = False  # Set to True to disable EBB input power voltage checks. Default: False

clip_to_page = True  # Clip plotting area to SVG document size. Default: True

min_gap = 0.008     # Automatic path joining threshold, inches. Default: 0.008
                    # If greater than zero, pen-up moves shorter than this distance
                    #   will be replaced by pen-down moves. Set negative to disable.
                    # Setting reordering to 4 (strict) will also disable path joining.

'''
Secondary control parameters:

Values below this point are configured only in this file, not through the user interface(s).
Please note that these values have been carefully chosen, and generally do not need to be 
adjusted in everyday use. That said, proceed with caution, and keep a backup copy.
'''

# Page size values typically do not need to be changed. They primarily affect viewpoint and centering.
# Measured in page pixelssteps.  Default printable area for iDraw is 300 x 218 mm

x_travel_default = 11.81 # iDraw V2, V3, SE/A4: X.    Default: 11.81 in (300 mm)
y_travel_default = 8.27  # iDraw V2, V3, SE/A4: Y.    Default:  8.27 in (210 mm)

x_travel_V3A3 = 16.93    # V3/A3 and SE/A3: X           Default: 16.93 in (430 mm)
y_travel_V3A3 = 11.69    # V3/A3 and SE/A3: Y           Default: 11.69 in (297 mm)

x_travel_V3XLX = 23.42   # iDraw V3 XLX: X            Default: 23.42 in (595 mm)
y_travel_V3XLX = 8.58    # iDraw V3 XLX: Y            Default:  8.58 in (218 mm)

x_travel_MiniKit = 6.30  # iDraw MiniKit: X           Default:  6.30 in (160 mm)
y_travel_MiniKit = 4.00  # iDraw MiniKit: Y           Default:  4.00 in (101.6 mm)

x_travel_SEA1 = 34.02    # iDraw SE/A1: X             Default: 34.02 in (864 mm)
y_travel_SEA1 = 23.39    # iDraw SE/A1: Y             Default: 23.39 in (594 mm)

x_travel_SEA2 = 23.39    # iDraw SE/A2: X             Default: 23.39 in (594 mm)
y_travel_SEA2 = 17.01    # iDraw SE/A2: Y             Default: 17.01 in (432 mm )

x_travel_V3B6 = 7.48     # iDraw V3/B6: X             Default: 7.48 in (190 mm)
y_travel_V3B6 = 5.51     # iDraw V3/B6: Y             Default: 5.51 in (140 mm)

x_travel_SEA0 = 46.85    # iDraw SE/A0: X             Default: 46.85 in (1189 mm)
y_travel_SEA0 = 33.11    # iDraw SE/A0: Y             Default: 33.11 in (841 mm )

native_res_factor = 1270.0  # Motor resolution calculation factor, steps per inch, and used in conversions. Default: 1016.0
# Note that resolution is defined along native (not X or Y) axes.
# Resolution is native_res_factor * sqrt(2) steps per inch in Low Resolution  (Approx 1437 steps per inch)
#       and 2 * native_res_factor * sqrt(2) steps per inch in High Resolution (Approx 2874 steps per inch)

max_step_rate = 24.995  # Maximum allowed motor step rate, in steps per millisecond.
# Note that 25 kHz is the absolute maximum step rate for the EBB.
# Movement commands faster than this are ignored; may result in a crash (loss of position control).
# We use a conservative value, to help prevent errors due to rounding.
# This value is normally used _for speed limit checking only_.

speed_lim_xy_lr = 110  # Maximum XY speed allowed when in Low Resolution mode, in inches per second.  Default: 15.000 Max: 17.3958
speed_lim_xy_hr = 110  # Maximum XY speed allowed when in High Resolution mode, in inches per second. Default: 8.6979, Max: 8.6979
# Do not increase these values above Max; they are derived from max_step_rate and the resolution.

max_step_dist_lr = 0.000696  # Maximum distance covered by 1 step in Low Res mode, rounded up, in inches. ~1/(1016 sqrt(2))
max_step_dist_hr = 0.000348  # Maximum distance covered by 1 step in Hi Res mode, rounded up, in inches.  ~1/(2032 sqrt(2))
# In planning trajectories, we skip movements shorter than these distances, likely to be < 1 step.

# const_speed_factor_lr = 0.25  # When in constant-speed mode, multiply the pen-down speed by this factor. Default: 0.25 for Low Res mode
# const_speed_factor_hr = 0.4  # When in constant-speed mode, multiply the pen-down speed by this factor. Default: 0.4 for Hi Res mode

start_pos_x = 0  # Parking position, inches. Default: 0
start_pos_y = 0  # Parking position, inches. Default: 0

# Acceleration & Deceleration rates:
accel_rate = 40.0    # Standard acceleration rate, inches per second squared
accel_rate_pu = 60.0  # Pen-up acceleration rate, inches per second squared

time_slice = 0.025  # Interval, in seconds, of when to update the motors. Default: time_slice = 0.025 (25 ms)

bounds_tolerance = 0.003  # Suppress warnings if bounds are exceeded by less than this distance (inches).

# Allow sufficiently short pen-up moves to be substituted with a pen-down move:
min_gap = 0.008  # Distance Threshold (inches). Default value: 0.008 inches; smaller than most pen lines.

# Servo motion limits, in units of (1/12 MHz), about 83 ns:
servo_max = 27831  # Highest allowed position; "100%" on the scale.    Default value: 25200 units, or 2.31 ms.
servo_min = 9855   # Lowest allowed position; "0%" on the scale.        Default value: 10800 units, or 0.818 ms.

# Note that previous versions of this configuration file used a wider range, 7500 - 28000, corresponding to a range of 625 us - 2333 us.
# The new limiting values are equivalent to 16%, 86% on that prior scale, giving a little less vertical range, but higher resolution.
# More importantly, it constrains the servo to within the travel ranges that they are typically calibrated, following best practice.

skip_voltage_check = False  # Set to True if you would like to disable EBB input power voltage checks. Default: False

clip_to_page = True  # Clip plotting area to SVG document size. Default: True

# the tolerance for determining when the bezier has been segmented enough to plot:
bezier_segmentation_tolerance = 0.02 / smoothness

# Tolerance for merging nearby vertices:
#  Larger values of segment_supersample_tolerance give smoother plotting along paths that
#  were created with too many vertices. A value of 0 will disable supersampling.
segment_supersample_tolerance = bezier_segmentation_tolerance / 16
curve_tolerance = 0.002 # Curve representation tolerance, inches. Default: 0.002 (0.05 mm)
button_interval = 0.05  # Minimum interval (s), for polling pause button. Default: 0.05 (50 ms)
servo_pin = 1  