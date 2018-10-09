"""
A package that provides submodules in order to perform polynomial chaos uncertanty analysis on complex dynamic systems.
"""

# initialize loggers for the whole package

import logging

global print_terminal, print_logfile
global console_logger, file_logger
print_terminal = False
print_logfile = False

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

file_logger.disabled = print_logfile
console_logger.disabled = print_terminal


def terminal_output(activated=True):
    global print_terminal
    print_terminal = activated
    console_logger.disabled = not print_terminal


def logfile_output(activated=True):
    global print_logfile
    print_logfile = activated
    file_logger.disabled = not print_logfile

