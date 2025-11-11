#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given two profiles extracted with extract_profile_from_output, prints the
differences between them.
"""

###########################################################################
# IMPORTS                                                                 #
###########################################################################
import getopt
import gzip
import pathlib
import sys
from prettytable import PrettyTable, MSWORD_FRIENDLY, PLAIN_COLUMNS


###########################################################################
# MISCELLANEOUS FUNCTIONS                                                 #
###########################################################################


def my_help():
    """Print the the command line usage help."""
    print("""Usage: diff_profiles.py [OPTION]... FILE1 FILE2

Given two profiles extracted with extract_profile_from_output, prints the
differences between them.

Options:
    -v, --VERBOSE      increment the output verbosity.
    -h, --help         display this help and exit.

Please, report bugs to <barrachi@uji.es>.
""")


def log(text):
    """Log a message to stderr."""
    sys.stderr.write(">>> %s\n" % text)


def error(text):
    """Report an error message and exit."""
    sys.stderr.write("ERROR: %s\n" % text)
    raise SystemExit(-1)


# Global command line parameters
INPUT_FILE_NAME = []
VERBOSE = 0
SCRIPT_PATH = pathlib.Path(__file__).parent.absolute()


def get_opts():
    """Read command line options."""
    global INPUT_FILE_NAME, VERBOSE
    optlist, args = getopt.getopt(sys.argv[1:],
                                  'hv',
                                  ['VERBOSE', 'help'])
    for opt, arg in optlist:
        if opt in ('-h', '--help'):
            my_help()
            raise SystemExit()
        elif opt in ('-v', '--verbosity'):
            VERBOSE = 1
    # Check required arguments
    if len(args) < 2:
        my_help()
        error("Two FILES are required")
    INPUT_FILE_NAME.append(args[0])
    INPUT_FILE_NAME.append(args[1])


###########################################################################
# APPLICATION SPECIFIC FUNCTIONS                                          #
###########################################################################
def file_to_dict(file):
    """Convert the CSV file to a dict."""
    _dict = {}
    total_time = 0.0
    # ncalls, tottime, percall, cumtime, percall, filename:lineno(function)
    file.readline()  # Ignore header
    for line in file.readlines():
        values = line.split(',', 5)
        for i in range(5):
            try:
                values[i] = int(values[i])
            except ValueError:
                try:
                    values[i] = float(values[i])
                except ValueError:
                    pass
        # Remove part of the path from values[5]
        splat_by_slash = values[5].split('/')
        values[5] = '/'.join(splat_by_slash[-3:])
        _dict[values[5]] = values[0:5]
        total_time += values[1]
    return _dict, total_time


def do_diff():
    """Do the diff"""
    data = []
    total_times = []
    print(f"Comparing '{INPUT_FILE_NAME[0]}' with '{INPUT_FILE_NAME[1]}'...")
    print()
    for i in range(2):
        file_open = open if INPUT_FILE_NAME[i][-3:] != ".gz" else gzip.open
        with file_open(INPUT_FILE_NAME[i], 'rt') as file:
            _dict, _total_time = file_to_dict(file)
            data.append(_dict)
            total_times.append(_total_time)
    t = PrettyTable(['ncalls', 'tottime', 'percall', 'cumtime', 'percall2', 'filename:lineno(function)'])
    t.set_style(PLAIN_COLUMNS)
    t.align = "r"
    t.align['filename:lineno(function)'] = "l"
    t.sortby = "tottime"
    t.reversesort = True
    common_keys = []
    for key, values1 in data[0].items():
        if key in data[1]:
            common_keys.append(key)
            values2 = data[1][key]
            try:
                values = [round(x - y, 3) for x, y in zip(values1, values2)]
            except TypeError:
                values = []
                for x, y in zip(values1, values2):
                    if isinstance(x, type(y)) == str:
                        x0, x1 = x.split('/')
                        y0, y1 = y.split('/')
                        # values.append(f"{int(x0)-int(y0)}/{int(x1)-int(y1)}")
                        values.append(int(x0) - int(y0))  # PrettyTable sort does not work with the previous version
                    else:
                        values.append(round(x - y, 3))
            if values == [0, 0.0, 0.0, 0.0, 0.0]:
                continue
            t.add_row([values[0], values[1], values[2], values[3], values[4], key])
    print(f"Differences between '{INPUT_FILE_NAME[0]}' and '{INPUT_FILE_NAME[1]}'")
    print(t)
    print()
    t.clear_rows()
    for common_key in common_keys:
        data[0].pop(common_key)
        data[1].pop(common_key)
    for i in range(2):
        if data[0]:
            print(f"Calls only in '{INPUT_FILE_NAME[i]}'")
            for key, values in data[i].items():
                t.add_row([values[0], values[1], values[2], values[3], values[4], key])
            print(t)
            print()
            t.clear_rows()
    print("Total times")
    t = PrettyTable([INPUT_FILE_NAME[0], INPUT_FILE_NAME[1]])
    t.align = "r"
    t.add_row([round(x, 3) for x in total_times])
    print(t)
    print()


def main():
    """Do the work (main function, called when not imported)."""
    get_opts()
    # Main part of the application
    do_diff()


if __name__ == "__main__":
    main()
