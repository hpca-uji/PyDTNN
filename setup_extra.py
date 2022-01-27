#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

"""
Extra command classes:
 + DevelopAndPostDevelop: develop command that can perform post development actions.
 + InstallAndPostInstall: install command that can perform post installation actions.
References:
 + /usr/lib64/python3.3/distutils/command/command_template
 + https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
"""

from setuptools.command.develop import develop
from setuptools.command.install import install


class DevelopAndPostDevelop(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION


class InstallAndPostInstall(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        # atexit.register(qtarmsim_post_install)
