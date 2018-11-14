# Author:  Analabha roy
# Contact: daneel@utexas.edu
from __future__ import division, print_function

"""
        Exact Dynamics for quantum
        spins and transverse fields with time-periodic drive

        * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        * Copyright (c) 2018 Analabha Roy (daneel@utexas.edu)
        *
        *This is free software: you can redistribute it and/or modify
        *it under the terms of version 2 of the GNU Lesser General
        *Public License as published by the Free Software Foundation.
        *Notes:
        *1. The initial state is currently hard coded to be the
        *classical ground  state of the Transverse Field Ising model
        * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""

__version__   = '0.1'
__author__    = 'Analabha Roy'
__credits__   = 'Department of Physics, The University of Burdwan'

__all__ = ["base"]
from base import *
