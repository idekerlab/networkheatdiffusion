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

SKIP_REASON = 'TEST_REMOTE_HEATDIFFUSION not set. Skipping real test'\
    ' of remote heat diffusion service'


@unittest.skipUnless(os.getenv('TEST_REMOTE_HEATDIFFUSION') is not None, SKIP_REASON)
class TestIntegrationHeatDiffusion(unittest.TestCase):

    TEST_DIR = os.path.dirname(__file__)

    TEST_NETWORK = os.path.join(TEST_DIR, 'data', 'testnetwork.cx')

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def get_dict_of_node_name_to_diffusion_rank(self, net_cx):
        """
        Parses network and builds a dictionary of node names as key and value
        is a tuple (rank, heat)

        NOTE: the rank and heat are not converted and are set as found in the
              network.

        :param net_cx:
        :return: dictionary where key is node name and value is a tuple (rank, heat)
        :rtype: dict
        """
        node_dict = dict()
        for node_id, node_obj in net_cx.get_nodes():
            n_attr = net_cx.get_node_attribute(node_id, 'diffusion_output_rank')
            if n_attr is None:
                continue
            rank = n_attr['v']
            n_attr = net_cx.get_node_attribute(node_id, 'diffusion_output_heat')
            self.assertIsNotNone(n_attr)
            node_dict[node_obj['n']] = (rank, n_attr['v'])
        return node_dict

    def test_diffusion_local_and_remote(self):
        net_cx = ndex2.create_nice_cx_from_file(TestIntegrationHeatDiffusion.TEST_NETWORK)
        diffuser = HeatDiffusion()
        diffuser.add_seed_nodes_by_node_name(net_cx, seed_nodes=['E', 'M'])
        local_diffuse_cx = diffuser.run_diffusion(net_cx)

        local_node_dict = self.get_dict_of_node_name_to_diffusion_rank(local_diffuse_cx)

        r2_cx = ndex2.create_nice_cx_from_file(TestIntegrationHeatDiffusion.TEST_NETWORK)
        diffuser.add_seed_nodes_by_node_name(r2_cx, seed_nodes=['E', 'M'])
        remote_diffuse_cx = diffuser.run_diffusion(r2_cx, via_service=True)

        remote_node_dict = self.get_dict_of_node_name_to_diffusion_rank(remote_diffuse_cx)

        # Couple issues found
        # there is some precision conversion issues going from str to float
        # when calling the remote service and the rank is indeterminate when
        # heat values are the same which is the case for the TEST_NETWORK
        # used here so we are testing rank only for the seeds 'E' and 'M'
        self.assertEqual(local_node_dict['E'][0],
                         remote_node_dict['E'][0])
        self.assertEqual(local_node_dict['M'][0],
                         remote_node_dict['M'][0])
        for node_name in local_node_dict.keys():
            self.assertAlmostEqual(local_node_dict[node_name][1],
                             remote_node_dict[node_name][1], places=10)


if __name__ == '__main__':
    sys.exit(unittest.main())
