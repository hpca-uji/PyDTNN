#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given an output file from benchmarks_CNN.py with the profile option
activated, prints a csv file with the profile information.
"""

###########################################################################
#  extract_profile_from_output.py                                         #
#  ---------------------------------------------------------------------  #
#    copyright            : (C) 2021 by Sergio Barrachina Mir             #
#    email                : barrachi@uji.es                               #
###########################################################################

###########################################################################
#                                                                         #
#  This program is free software; you can redistribute it and/or modify   #
#  it under the terms of the GNU General Public License as published by   #
#  the Free Software Foundation; either version 2 of the License, or      #
#  (at your option) any later version.                                    #
#                                                                         #
#  This program is distributed in the hope that it will be useful, but    #
#  WITHOUT ANY WARRANTY; without even the implied warranty of             #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU      #
#  General Public License for more details.                               #
#                                                                         #
###########################################################################

###########################################################################
# IMPORTS                                                                 #
###########################################################################
import re
import sys
import getopt
import pathlib
import gzip
import tempfile


###########################################################################
# MISCELLANEOUS FUNCTIONS                                                 #
###########################################################################
def my_help():
    """Print the the command line usage help."""
    print("""Usage: extract_profile_from_output.py [OPTION]... FILE

Given an output file from benchmarks_CNN.py with the profile option
activated, prints a csv file with the profile information.

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
    sys.exit(-1)


# Global command line parameters
INPUT_FILE_NAME = None
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
            sys.exit()
        elif opt in ('-v', '--verbosity'):
            VERBOSE = 1
    # Check required arguments
    if len(args) == 0:
        my_help()
        error("At least a FILE is required")
    INPUT_FILE_NAME = args[0]


###########################################################################
# APPLICATION SPECIFIC FUNCTIONS                                          #
###########################################################################
def print_profile_from_file(file):
    """Do print the profile part from file."""
    first_line_re = re.compile("ncalls *tottime")
    in_profile_section = False
    for line in file.readlines():
        if first_line_re.search(line):
            in_profile_section = True
        if in_profile_section:
            line = line.strip()
            if line == "":
                return
            line = line.split(None, 5)
            print(",".join(line))


def print_profile():
    """Get the file and print the profile"""
    if INPUT_FILE_NAME is None:
        with sys.stdin as file:
            print_profile_from_file(file)
    else:
        file_open = open if INPUT_FILE_NAME[-3:] != ".gz" else gzip.open
        with file_open(INPUT_FILE_NAME, 'rt') as file:
            print_profile_from_file(file)


def main():
    """Do the work (main function, called when not imported)."""
    get_opts()
    # Main part of the application
    print_profile()


if __name__ == "__main__":
    main()
