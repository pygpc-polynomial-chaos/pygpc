"""
A package that provides submodules in order to perform polynomial chaos uncertainty analysis on complex dynamic systems.
"""

# initialize logger
import logging

file_logger = logging.getLogger('gPC')
file_logger.setLevel(logging.DEBUG)
file_logger_handler = logging.FileHandler('gPC.log')
file_logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
file_logger_handler.setFormatter(file_logger_formatter)
file_logger.addHandler(file_logger_handler)

console_logger = logging.getLogger('gPC_console_output')
console_logger.setLevel(logging.DEBUG)
console_logger_handler = logging.StreamHandler()
console_logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_logger_handler.setFormatter(console_logger_formatter)
console_logger.addHandler(console_logger_handler)

file_logger.disabled = False
console_logger.disabled = False


def activate_terminal_output():
    console_logger.disabled = False


def activate_logfile_output():
    file_logger.disabled = False


def deactivate_terminal_output():
    console_logger.disabled = True


def deactivate_logfile_output():
    file_logger.disabled = True


def iprint(message, verbose=True, tab=None):
    """
    Function that prints out a message over the python logging module

    iprint(message, verbose=True)

    Parameters
    ----------
    message: string
        string to print in standard output
    verbose: bool, optional, default=True
        determines if string is printed out
    """
    
    if tab:
        message = '\t'*tab + message
    console_logger.info(message)


def wprint(message, verbose=True, tab=None):
    """
    Function that prints out a warning message over the python logging module

    wprint(message, verbose=True)

    Parameters
    ----------
    message: string
        string to print in standard output
    verbose: bool, optional, default=True
        determines if string is printed out
    """

    if tab:
        message = '\t'*tab + message
    console_logger.warning(message)


import misc
import gpc
import grid
import postproc
import quad
import reg
import rw
import testfun
import vis
