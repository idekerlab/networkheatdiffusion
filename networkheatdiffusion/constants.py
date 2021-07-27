# -*- coding: utf-8 -*-

import numpy

"""
Contains constants used by the Network Heat Diffusion
"""

DEFAULT_SERVICE_ENDPOINT = 'http://v3.heat-diffusion.cytoscape.io'
"""
Default diffusion service REST endpoint
"""

DEFAULT_INPUT = 'diffusion_input'
"""
Default diffusion input attribute name
"""

DEFAULT_OUTPUT_PREFIX = 'diffusion_output'
"""
Default diffusion output attribute prefix
"""

DEFAULT_HEAT_SUFFIX = '_heat'
"""
Default heat attribute suffix
"""

DEFAULT_RANK_SUFFIX = '_rank'
"""
Default rank attribute suffix
"""

DEFAULT_HEAT = DEFAULT_OUTPUT_PREFIX + DEFAULT_HEAT_SUFFIX
"""
Default heat attribute name
"""

DEFAULT_RANK = DEFAULT_OUTPUT_PREFIX + DEFAULT_RANK_SUFFIX
"""
Default rank attribute name
"""

DEFAULT_DATA_TYPE = numpy.float64
"""
Default data type for :py:module:`scipy` and :py:mod:`numpy` operations
"""
