from setuptools import setup, find_packages

setup(name='pygpc',
      version='0.1',
      description='polynomial chaos expansion',
      author='Konstantin Weise',
      author_email='konstantin.weise@tu-ilmenau.de',
      license='GPL3',
      packages=['pygpc', 'pygpc/testfuns'],
      zip_safe=False)
