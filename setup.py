import argparse
import os
import numpy as np
from setuptools import setup, find_packages, Extension


# Pygpc software framework for uncertainty and sensitivity
# analysis of complex systems. See also:
# https://github.com/pygpc-polynomial-chaos/pygpc
#
# Copyright (C) 2017-2020 the original author (Konstantin Weise),
# the Max-Planck-Institute for Human Cognitive Brain Sciences ("MPI CBS")
# and contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>


openmp_compile_args = ['-fopenmp']
openmp_link_args = ['-lgomp']
pygpc_extensions_src_file_path = [os.path.join('pckg', 'pygpc_extensions',
                                               'src', 'pygpc_extensions.cpp')]
pygpc_extensions_include_path = [os.path.join('pckg', 'pygpc_extensions',
                                              'include'), np.get_include()]

extensions = [Extension('pygpc.pygpc_extensions',
                        sources=pygpc_extensions_src_file_path,
                        include_dirs=pygpc_extensions_include_path,
                        extra_compile_args=openmp_compile_args,
                        extra_link_args=openmp_link_args)]


setup(name='pygpc',
      version='0.2.8',
      description='A sensitivity and uncertainty analysis toolbox for Python',
      author='Konstantin Weise',
      author_email='kweise@cbs.mpg.de',
      license='GPL3',
      packages=find_packages(exclude=['tests',
                                      'tests.*',
                                      'templates',
                                      'templates.*']),
      install_requires=['scipy>=1.0.0',
                        'numpy>=1.16.4',
                        'fastmat>=0.1.2.post1',
                        'scikit-learn>=0.19.1',
                        'h5py>=2.9.0'],
      ext_modules=extensions,
      package_data={'pygpc': ['*.so', '*.dll', '*.dylib']},
      project_urls={
        "Documentation": "https://pygpc.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/pygpc-polynomial-chaos/pygpc"},
      zip_safe=False,
      include_package_data=True)
