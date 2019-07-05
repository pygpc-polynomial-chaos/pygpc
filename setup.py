from setuptools import setup, find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


# pygpc software framework for uncertainty and sensitivity
# analysis of complex systems. See also:
# https://github.com/konstantinweise/pygpc
#
# Copyright (C) 2017-2019 the original author (Konstantin Weise),
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

ext_modules = [
    Extension(
        name="calc_gpc_matrix_cpu",
        sources=['./pckg/extensions/calc_gpc_matrix_cpu/calc_gpc_matrix_cpu.pyx'],
        libraries=['gpc'],
        library_dirs=['./pckg/extensions/calc_gpc_matrix_cpu'],
        include_dirs=['./pckg/extensions/calc_gpc_matrix_cpu']
    )
]


setup(name='pygpc',
      version='0.1',
      description='A Sensitivity and uncertainty analysis toolbox for Python',
      author='Konstantin Weise',
      author_email='konstantin.weise@tu-ilmenau.de',
      license='GPL3',
      packages=find_packages(),
      install_requires=['scipy',
                        'fastmat',
                        'sklearn',
                        'h5py',
                        'matplotlib',
                        'dispy',
                        ],
      zip_safe=False,
      ext_modules=cythonize(ext_modules))
