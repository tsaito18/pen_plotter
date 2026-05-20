from copy import deepcopy
from collections import namedtuple
from optparse import OptionGroup

#=== Consider refactoring to `plotink` or another utilities package
ArgumentAttrs = namedtuple('ArgumentAttrs', ['flags', 'type', 'kwargs', 'choose_default'])

def add_options(opt_parser, option_list, config):
    ''' Same as `add_arguments`, below, with the additional restriction that
    `opt_parser` must have been created with `option_class=inkex.InkOption` so
    that type `'inkbool' works.`.'''
    for option in option_list:

        kwargs = deepcopy(arguments[option].kwargs)
        kwargs['type'] = 'inkbool' if arguments[option].type == bool else arguments[option].type
        kwargs['default'] = arguments[option].choose_default(config)

        opt_parser.add_option(
            *arguments[option].flags,
            **kwargs)

def add_arguments(arg_parser, argument_list, config):
    ''' Add the arguments in `argument_list` to `arg_parser`.

    `arg_parser`: an object that can take the `add_argument` method from the
        `argparse` modulele, e.g. `argparse.ArgumentParser` or an `argument_group`.
    `argument_list`: a list of strings, each string being a key in the `arguments` dict, below
    `config`: a dictionary to be used as a parameter in `ArgumentAttrs.choose_default()`'''
    for argument in argument_list:
        kwargs = deepcopy(arguments[argument].kwargs)

        if arguments[argument].type == bool:
            # some special stuff so the argument doesn't require an explicit value
            kwargs['action'] = "store_const"
            kwargs['const'] = True
        else:
            kwargs['type'] = arguments[argument].type

        kwargs['default'] = arguments[argument].choose_default(config)

        arg_parser.add_argument(
            *arguments[argument].flags,
            **kwargs)
#===

def core_options(parser, config):
    options = OptionGroup(parser, "Core Options")

    option_names = ["font_face", "font_option", "letter_spacing", "word_spacing", "enable_defects",
                    "leading_var", "baseline_var", "indent_var", "kern_var", "size_var"]
    add_options(options, option_names, config)

    return options

def extra_options(parser, config):
    options = OptionGroup(parser, "Additional Options")

    option_names = ["mode", "sample_text", "util_mode", "preserve_text", "rand_seed"]
    add_options(options, option_names, config)

    return options

arguments = {
    "font_face": ArgumentAttrs(
        ["-F", "--font_face"],
        str,
        { "dest": "font_face",
          "help": "Specify font name or path to use" },
        lambda config: config.get("font_face")),
    "font_option": ArgumentAttrs(
        ["--font_option"],
        str,
        { "dest": "font_option",
          "help": "Default fallback font or 'other' (GUI ONLY)"},
        lambda _: None),
    "letter_spacing": ArgumentAttrs(
        ["-L", "--letter_spacing"],
        int,
        { "dest": "letter_spacing",
          "help": "Override letter spacing "},
        lambda config: config.get("letter_spacing")),
    "word_spacing": ArgumentAttrs(
        ["-W", "--word_spacing"],
        int,
        { "dest": "word_spacing",
          "help": "Override word spacing " },
        lambda config: config.get("word_spacing")),
    "enable_defects": ArgumentAttrs(
        ["-D", "--enable_defects"],
        bool,
        { "dest": "enable_defects",
          "help": "Enable Handwriting Defects"},
        lambda config: config.get("enable_defects")),
    "leading_var": ArgumentAttrs(
        ["-l", "--leading_var"],
        int,
        { "dest": "leading_var",
          "help": "Variation in line spacing (leading)" },
        lambda config: config.get("leading_var")),
    "baseline_var": ArgumentAttrs(
        ["-B", "--baseline_var"],
        int,
        { "dest": "baseline_var",
          "help": "Variation in baseline Jitter" },
        lambda config: config.get("baseline_var")),
    "indent_var": ArgumentAttrs(
        ["-I", "--indent_var"],
        int,
        { "dest": "indent_var",
          "help": "Variation in indent "},
        lambda config: config.get("indent_var")),
    "kern_var": ArgumentAttrs(
        ["-K", "--kern_var"],
        int,
        { "dest": "kern_var",
          "help": "Variation in letter kerning "},
        lambda config: config.get("kern_var")),
    "size_var": ArgumentAttrs(
        ["-S", "--size_var"],
        int,
        { "dest": "size_var",
          "help": "Variation in font size " },
        lambda config: config.get("size_var")),
    "mode": ArgumentAttrs(
        ["-m", "--mode"],
        str,
        { "dest": "mode",
          "help": "General mode of operation: render, glyph_table, or font_table"},
        lambda config: config.get("mode")),
    "sample_text": ArgumentAttrs(
        ["-t", "--sample_text"],
        str,
        { "dest": "sample_text",
          "help": "Text to use for font table"},
        lambda config: config.get("sample_text")),
    "util_mode": ArgumentAttrs(
        ["--util_mode"],
        str,
        { "dest": "util_mode",
          "help": "Utility mode (GUI ONLY)"},
        lambda _: "font_table"),
    "preserve_text": ArgumentAttrs(
        ["-p", "--preserve_text"],
        bool,
        { "dest": "preserve_text",
          "help": "Preserve original text"},
        lambda config: config.get("preserve_text")),
    "rand_seed": ArgumentAttrs(
        ["-r", "--rand_seed"],
        int,
        { "dest": "rand_seed",
          "help": "Random seed. Choose 1 to use time."},
        lambda config: config.get("rand_seed")),
}
