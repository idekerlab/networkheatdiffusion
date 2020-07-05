#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_heatdiffusionclient
----------------------------------

Tests for `networkheatdiffusion` module.
"""

import os
import sys
import unittest
import ndex2

from networkheatdiffusion import HeatDiffusion


class TestHeatDiffusion(unittest.TestCase):

    TEST_DIR = os.path.dirname(__file__)

    TEST_NETWORK = os.path.join(TEST_DIR, 'data', 'testnetwork.cx')

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_diffusion(self):
        net_cx = ndex2.create_nice_cx_from_file(TestHeatDiffusion.TEST_NETWORK)
        diffuser = HeatDiffusion()
        diffuser.add_seed_nodes_by_node_name(net_cx, seed_nodes=['E', 'M'])
        res_cx = diffuser.run_diffusion(net_cx)
        for node_id, node_obj in res_cx.get_nodes():
            n_attr = res_cx.get_node_attribute(node_id, 'diffusion_output_rank')
            self.assertIsNotNone(n_attr)
            if n_attr['v'] == 0:
                self.assertEqual('E', node_obj['n'])
            n_attr = res_cx.get_node_attribute(node_id, 'diffusion_output_heat')
            self.assertIsNotNone(n_attr)



if __name__ == '__main__':
    sys.exit(unittest.main())
