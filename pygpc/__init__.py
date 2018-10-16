"""
A package that provides submodules in order to perform polynomial chaos uncertainty analysis on complex dynamic systems.
"""

# initialize loggers for the whole package

import logging


# initialize logger
file_logger = logging.getLogger('gPC')
file_logger.setLevel(logging.DEBUG)
file_logger_handler = logging.FileHandler('gPC.log')
file_logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
file_logger_handler.setFormatter(file_logger_formatter)
file_logger.addHandler(file_logger_handler)

console_logger = logging.getLogger('gPC_console_output')
console_logger.setLevel(logging.DEBUG)
console__logger_handler = logging.StreamHandler()
console_logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console__logger_handler.setFormatter(console_logger_formatter)
console_logger.addHandler(console__logger_handler)

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
