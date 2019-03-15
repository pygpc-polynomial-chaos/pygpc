from setuptools import setup, find_packages

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
                        'pytest',
                        'numpy'],
      zip_safe=False)
