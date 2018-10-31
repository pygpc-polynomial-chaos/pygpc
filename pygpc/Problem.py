#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from collections import namedtuple

RandomParameter = namedtuple("RandomParameter", "pdftype pdfshape limits")

class Problem:
    """
        Data wrapper for the gpc problem.
    """
    def __init__(self, modelClass, parametersDict):
        assert( isinstance(parametersDict, OrderedDict) )

        self.modelClass = modelClass
        self.parameters = parametersDict
