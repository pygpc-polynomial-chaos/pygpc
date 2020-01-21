from setuptools import setup, find_packages
from distutils.extension import Extension


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


# parser = argparse.ArgumentParser('Install pygpc with or without multicore and/or gpu support')
# parser.add_argument('--enable-openmp', type=bool, default=False)
# args = parser.parse_args()
# 
# 
# if(args.enable-openmp):
#     print('OpenMP')


setup(name='pygpc',
      version='0.2.6.post2',
      description='A sensitivity and uncertainty analysis toolbox for Python',
      author='Konstantin Weise',
      author_email='kweise@cbs.mpg.de',
      license='GPL3',
      cmdclass={
          'install': InstallCommand
      },
      packages=find_packages(exclude=['tests', 'tests.*', 'templates', 'templates.*', 'tutorials', 'tutorials.*']),
      install_requires=['scipy>=1.0.0',
                        'numpy>=1.16.4',
                        'fastmat>=0.1.2.post1',
                        'scikit-learn>=0.19.1',
                        'h5py>=2.9.0',
                        'dispy>=4.9.0'],
      project_urls={
        "Documentation": "https://pygpc.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/pygpc-polynomial-chaos/pygpc"},
      zip_safe=False,
      include_package_data=True)
