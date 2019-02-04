from setuptools import setup, find_packages

setup(name='PySense',
      version='0.1',
      description='Sensitivity and uncertainty analysis toolbox for Python',
      author='Konstantin Weise',
      author_email='konstantin.weise@tu-ilmenau.de',
      license='GPL3',
      packages=find_packages(),
      install_requires=['numpy',
                        'scipy',
                        'sklearn',
                        'h5py',
                        'pyyaml',
                        'dill',
                        'matplotlib',
                        'dispy'],
      zip_safe=False)
