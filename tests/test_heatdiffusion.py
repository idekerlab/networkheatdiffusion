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
import networkx
import numpy as np

from networkheatdiffusion import HeatDiffusion
from networkheatdiffusion import HeatDiffusionError


class TestHeatDiffusion(unittest.TestCase):

    TEST_DIR = os.path.dirname(__file__)

    TEST_NETWORK = os.path.join(TEST_DIR, 'data', 'testnetwork.cx')

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_diffusion_constructor(self):
        diffuser = HeatDiffusion()
        self.assertEqual(HeatDiffusion.DEFAULT_SERVICE_ENDPOINT,
                         diffuser._service_endpoint)
        self.assertEqual(360, diffuser._connect_timeout)

        diffuser = HeatDiffusion(service_endpoint='http://foo.com',
                                 connect_timeout=10)
        self.assertEqual('http://foo.com', diffuser._service_endpoint)
        self.assertEqual(10, diffuser._connect_timeout)

    def test_create_sparse_matrix(self):
        my_net = networkx.MultiGraph()
        my_net.add_nodes_from([1, 2, 3])
        my_net.add_edges_from([(1, 2),
                               (1, 3)])

        diffuser = HeatDiffusion()
        res_array = diffuser._create_sparse_matrix(my_net).toarray()

        self.assertTrue(np.array_equal(np.array([2, -1, -1]),
                                       res_array[0]))

        self.assertTrue(np.array_equal(np.array([-1, 1, 0]),
                                       res_array[1]))

        self.assertTrue(np.array_equal(np.array([-1, 0, 1]),
                                       res_array[2]))

        res_array = diffuser._create_sparse_matrix(my_net,
                                                   normalize=True).toarray()
        print(res_array.shape)
        self.assertTrue(np.isclose(np.array([1, -0.70710678, -0.70710678]),
                                   res_array[0]).all())

        self.assertTrue(np.isclose(np.array([-0.70710678, 1, 0]),
                                   res_array[1]).all())

        self.assertTrue(np.isclose(np.array([-0.70710678, 0, 1]),
                                   res_array[2]).all())

    def test_find_heat_no_heat_key(self):
        my_net = networkx.MultiGraph()
        my_net.add_nodes_from([1, 2, 3])

        diffuser = HeatDiffusion()
        try:
            diffuser._find_heat(my_net, 'heat')
            self.fail('Expected HeatDiffusionError')
        except HeatDiffusionError as he:
            self.assertEqual('No input heat found', str(he))

    def test_find_heat(self):
        my_net = networkx.MultiGraph()
        my_net.add_nodes_from([1, 2, 3])
        node_attrs = {1: {'heat': 1}, 2: {'heat': 2}}
        networkx.set_node_attributes(my_net, node_attrs)
        diffuser = HeatDiffusion()
        res = diffuser._find_heat(my_net, 'heat')
        self.assertTrue(np.array_equal(np.array([1, 2, 0]), res))

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

    def test_add_heat(self):
        my_net = networkx.MultiGraph()
        my_net.add_nodes_from([1, 2, 3])

        diffuser = HeatDiffusion()
        node_heat, node_rank = diffuser._add_heat(my_net,
                                                  np.array([2, 3, 1]))
        self.assertEqual({1: 2, 2: 3, 3: 1}, node_heat)
        self.assertEqual({2: 0, 1: 1, 3: 2}, node_rank)

    def test_convert_attribute_values_to_strings_where_change_needed(self):
        net_cx = ndex2.nice_cx_network.NiceCXNetwork()
        node_id = net_cx.create_node('foo')
        net_cx.add_node_attribute(property_of=node_id, name='attrib',
                                  values=1, type='integer')
        diffuser = HeatDiffusion()
        res = diffuser._convert_attribute_values_to_strings(net_cx.to_cx())
        val = None
        for aspect in res:
            if 'nodeAttributes' in aspect:
                val = aspect['nodeAttributes'][0]['v']
        self.assertTrue(isinstance(val, str))
        self.assertEqual('1', val)

    def test_convert_attribute_values_to_strings_nochange_needed(self):
        net_cx = ndex2.nice_cx_network.NiceCXNetwork()
        node_id = net_cx.create_node('foo')
        net_cx.add_node_attribute(property_of=node_id, name='attrib',
                                  values='foo')
        diffuser = HeatDiffusion()
        res = diffuser._convert_attribute_values_to_strings(net_cx.to_cx())
        val = None
        for aspect in res:
            if 'nodeAttributes' in aspect:
                val = aspect['nodeAttributes'][0]['v']
        self.assertTrue(isinstance(val, str))
        self.assertEqual('foo', val)

    def test_build_post_url(self):
        diffuser = HeatDiffusion()
        # try with no arguments
        res = diffuser._build_post_url()
        self.assertEqual({}, res)

        # try with normalize_laplacian True
        res = diffuser._build_post_url(normalize_laplacian=True)
        self.assertEqual({'normalize_laplacian': True}, res)

        # try with all normalize_laplacian False
        res = diffuser._build_post_url(time_param=0.5,
                                       normalize_laplacian=False,
                                       input_col_name='inputcol',
                                       output_prefix='outputcol')
        self.assertEqual({'time': 0.5,
                          'input_attribute_name': 'inputcol',
                          'output_attribute_name': 'outputcol'}, res)



if __name__ == '__main__':
    sys.exit(unittest.main())
