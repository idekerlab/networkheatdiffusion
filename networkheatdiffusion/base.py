# -*- coding: utf-8 -*-

import logging
import requests
import networkx
import numpy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
from ndex2.nice_cx_network import NiceCXNetwork
from ndex2.nice_cx_network import DefaultNetworkXFactory

LOGGER = logging.getLogger(__name__)


class HeatDiffusionError(Exception):
    """
    Base Exception for errors in networkheatdiffusion
    """
    pass


class HeatDiffusion(object):
    """
    Runs heat diffusion on remote service
    """

    DEFAULT_SERVICE_ENDPOINT = 'http://v3.heat-diffusion.cytoscape.io'
    DEFAULT_INPUT = 'diffusion_input'
    DEFAULT_OUTPUT_PREFIX = 'diffusion_output'
    DEFAULT_HEAT_SUFFIX = '_heat'
    DEFAULT_RANK_SUFFIX = '_rank'
    DEFAULT_HEAT = DEFAULT_OUTPUT_PREFIX + DEFAULT_HEAT_SUFFIX
    DEFAULT_RANK = DEFAULT_OUTPUT_PREFIX + DEFAULT_RANK_SUFFIX

    def __init__(self, service_endpoint=None, connect_timeout=360):
        """
        Constructor
        :param service_endpoint: URL for making requests to diffusion service
        :type service_endpoint: str
        :param connect_timeout: timeout in seconds to wait for connection to service
        :type connect_timeout: int
        """
        self._connect_timeout = connect_timeout
        self._service_endpoint = service_endpoint
        if self._service_endpoint is None:
            self._service_endpoint = HeatDiffusion\
                .DEFAULT_SERVICE_ENDPOINT
        else:
            LOGGER.debug('Using custom service endpoint: ' +
                         self._service_endpoint)

    def _create_sparse_matrix(self, network, normalize=False):
        """

        :param network:
        :param normalize:
        :return:
        """
        if normalize:
            return csc_matrix(networkx.normalized_laplacian_matrix(network))
        else:
            return csc_matrix(networkx.laplacian_matrix(network))

    def _diffuse(self, matrix, heat_array, time):
        """

        :param matrix:
        :param heat_array:
        :param time:
        :return:
        """
        return expm_multiply(-matrix, heat_array,
                          start=0, stop=time,
                          endpoint=True)[-1]

    def _find_heat(self, network, heat_key):
        """
        Gets node heat values from 'network' passed in

        :param network: network with nodes that contain 'heat_key' attribute
        :type network: :py:class:`networkx.Graph`
        :param heat_key: name of heat key ie diffusion_input
        :type heat_key: str
        :return: array of heat values in order of index values
                 of nodes in network
        :rtype: :py:class:`numpy.ndarray`
        """
        heat_list = []
        found_heat = False
        for node_id in network.nodes():
            if heat_key in network.nodes[node_id]:
                found_heat = True
            heat_list.append(network.nodes[node_id].get(heat_key, 0))
        if not found_heat:
            raise HeatDiffusionError('No input heat found')
        return numpy.array(heat_list)

    def _add_heat(self, network, heat_array,
                  correct_rank=False):
        """
        Given a 'network' and an array of heats in 'heat_array' this
        method returns a :py:class:`tuple` with two :py:class:`dict`
        objects.

        The first contains node heats with key set to
        node id and value being the heat.

        The second contains node ranks with key set to
        node id and value being the rank where 0 is best
        rank and set to node with largest heat value.

        :param network:
        :type network: :py:class:`networkx.Graph`
        :param heat_array: array of network node heats ordered by nodes
                           in 'network'
        :type heat_array: :py:class:`numpy.ndarray`
        :param correct_rank: If True, multiple nodes that have same heat
                             will have same rank.
        :type correct_rank: bool
        :return: (node heat as :py:class:`dict` with node id as key,
                  node rank as :py:class:`dict` with node id as key)
        :rtype: tuple
        """
        node_heat = {node_id: heat_array[i] for i, node_id in enumerate(network.nodes())}
        sorted_nodes = sorted(node_heat.items(), key=lambda x: x[1], reverse=True)

        # this is a little correction that differs from REST service
        # where if multiple nodes have same heat value they are given
        # the same rank
        if correct_rank is True:
            node_rank = dict()
            rank = 0
            previous_heat = None
            for _, (node_id, heat) in enumerate(sorted_nodes):
                if previous_heat is not None:
                    if heat < previous_heat:
                        rank += 1
                node_rank[node_id] = rank

                previous_heat = heat
        else:
            node_rank = {node_id: i for i, (node_id, _) in enumerate(sorted_nodes)}

        return node_heat, node_rank

    def _add_diffusion_dict_to_network(self, cxnetwork, node_heat, node_rank,
                                       heat_col_name=DEFAULT_HEAT,
                                       rank_col_name=DEFAULT_RANK):
        """
        Adds heat and rank as node attributes to 'cxnetwork' network

        :param cxnetwork:
        :type cxnetwork: :py:class:`ndex2.nice_cx_network.NiceCXNetwork`
        :param node_heat: :py:class:`dict` where key is node id and value is
                          heat for that node
        :type node_heat: dict
        :param node_rank: :py:class:`dict` where key is node id and value is
                          rank for that node
        :type node_rank: dict
        :return: 'cxnetwork' passed in
        """
        for node_id, node_obj in cxnetwork.get_nodes():
            if node_id in node_heat:
                cxnetwork.add_node_attribute(property_of=node_id,
                                             name=heat_col_name,
                                             values=node_heat[node_id],
                                             type='double',
                                             overwrite=True)
            if node_id in node_rank:
                cxnetwork.add_node_attribute(property_of=node_id,
                                             name=rank_col_name,
                                             values=node_rank[node_id],
                                             type='integer',
                                             overwrite=True)
        return cxnetwork

    def run_diffusion(self, cxnetwork, time_param=0.1,
                      normalize_laplacian=False,
                      input_col_name=DEFAULT_INPUT,
                      output_prefix=DEFAULT_OUTPUT_PREFIX,
                      correct_rank=False,
                      via_service=False,
                      service_read_timeout=360):
        """
        Runs diffusion annotating the 'cxnetwork' passed in with
        new node attributes 'outputprefix'_heat and 'output_prefix'_rank
        added to 'cxnetwork' in place.

        :param cxnetwork: network to run diffusion on
        :type cxnetwork: :py:class:`~ndex2.nice_cx_network.NiceCXNetwork`
        :param time_param: stop time passed to :py:func:`scipy.sparse.linalg.expm_multiply`
        :type time_param: int
        :param normalize_laplacian: If `True`, will create a normalized
                                    laplacian matrix for diffusion.
                                    `None` denotes default of `False`
        :type normalize_laplacian: bool
        :param input_col_name: Name of node attribute that contains
                               diffusion heat inputs which should
                               be double values between `0.0` and
                               `1.0` where `1.0` is maximum heat
                               (seed) and 0.0 is minimum.
                               `None` denotes default
                               input column name of `diffusion_input`
        :type input_col_name: str
        :param output_prefix: Prefix name for diffusion output attached
                              to each node. Each
                              node will get <PREFIX>_rank and
                              <PREFIX>_heat. `None` denotes
                              default prefix of `diffusion_output`
        :type output_prefix: str
        :param correct_rank: If True, multiple nodes that have same heat
                             will have same rank. NOTE: This only works
                             with local invocation of diffusion at the moment
                             and will raise a
                             :py:class:`~networkheatdiffusion.base.HeatDiffusionError`
                             if set to True and invoked with `via_service` set to `True`
        :type correct_rank: bool
        :param via_service: if `True` run diffusion via remote service
        :param service_read_timeout: Seconds to wait for a response
                                     from service. Only used when 'via_service' is
                                     set to `True`
        :type service_read_timeout: int
        :raises HeatDiffusionError: If there is an error
        :return: network passed in with diffusion columns added
        :rtype: `:py:class:`~ndex2.nice_cx_network.NiceCXNetwork`
        """
        if via_service is not None and via_service is True:
            if correct_rank is True:
                raise HeatDiffusionError('correct_rank flag only works with local invocation'
                                         'of diffusion')
            return self._run_diffusion_via_service(cxnetwork, time_param=time_param,
                                                   normalize_laplacian=normalize_laplacian,
                                                   input_col_name=input_col_name,
                                                   output_prefix=output_prefix,
                                                   service_read_timeout=service_read_timeout)
        netx_fac = DefaultNetworkXFactory()
        netx_graph = netx_fac.get_graph(cxnetwork, networkx_graph=networkx.MultiGraph())
        matrix = self._create_sparse_matrix(netx_graph, normalize_laplacian)
        heat_array = self._find_heat(netx_graph, input_col_name)
        diffused_heat_array = self._diffuse(matrix, heat_array, time_param)
        node_heat, node_rank = self._add_heat(netx_graph,
                                              diffused_heat_array, correct_rank=correct_rank)
        return self._add_diffusion_dict_to_network(cxnetwork, node_heat, node_rank)

    def _run_diffusion_via_service(self, cxnetwork, time_param=None,
                                   normalize_laplacian=None,
                                   input_col_name=None,
                                   output_prefix=None,
                                   service_read_timeout=360):
        """
        Runs diffusion on 'network' passed in, storing results in two new node
        attributes named by the values of 'diffusion_output_rank_col_name' and
        'diffusion_output_heat_col_name' parameters passed into this method.

        To run properly, the network must have `1` or more
        nodes set as seed nodes.

        This can be done by adding a node attribute
        with name set to the value of 'diffusion_input_col_name' with a
        double value of `1.0`

        :param cxnetwork: Network to run diffusion on
        :type cxnetwork: `:py:class:`~ndex2.nice_cx_network.NiceCXNetwork`
        :param time_param: time parameter for diffusion.
                           `None` denotes default of `0.5`.
        :type time_param: float
        :param normalize_laplacian: If `True`, will create a normalized
                                    laplacian matrix for diffusion.
                                    `None` denotes default of `False`
        :type normalize_laplacian: bool
        :param input_col_name: Name of node attribute that contains
                               diffusion heat inputs which should
                               be double values between `0.0` and
                               `1.0` where `1.0` is maximum heat
                               (seed) and 0.0 is minimum.
                               `None` denotes default
                               input column name of `diffusion_input`
        :param output_prefix: Prefix name for diffusion output attached
                              to each node. Each
                              node will get <PREFIX>_rank and
                              <PREFIX>_heat. `None` denotes
                              default prefix of `diffusion_output`
        :param service_read_timeout: Seconds to wait for a response
                                     from service
        :type service_read_timeout: int
        :return: network passed in with diffusion columns added
        :rtype: `:py:class:`~ndex2.nice_cx_network.NiceCXNetwork`
        """
        if cxnetwork is None:
            raise HeatDiffusionError('Input cxnetwork cannot be None')
        if isinstance(cxnetwork, NiceCXNetwork) is False:
            LOGGER.warning('cxnetwork passed in is '
                           'not of type NiceCXNetwork. Code may fail')

        LOGGER.debug('Converting CX to list of dictionaries')
        payload = cxnetwork.to_cx()

        self._convert_attribute_values_to_strings(payload)

        params = self._build_post_url(time_param=time_param,
                                      normalize_laplacian=normalize_laplacian,
                                      input_col_name=input_col_name,
                                      output_prefix=output_prefix)
        LOGGER.debug('Submitting request to ' + self._service_endpoint +
                     ' with params: ' + str(params))
        resp = requests.post(self._service_endpoint,
                             params=params,
                             json=payload,
                             timeout=(self._connect_timeout,
                                      service_read_timeout))
        LOGGER.debug('Received: ' + str(resp.status_code) +
                     ' response from service')
        if resp.status_code != 200:
            raise HeatDiffusionError('Received error code: ' +
                                     str(resp.status_code) +
                                     ' : (' + str(resp.text) +
                                     ') from call to diffusion service: ' +
                                     self._service_endpoint)

        return self._append_diffusion_result_to_network(cxnetwork, resp.json(),
                                                        output_prefix)

    @staticmethod
    def _convert_attribute_values_to_strings(cx_as_list_of_dictionaries):
        """
        The remote diffusion service expects all values in all attributes
        aspects to be string values so this method converts those values
        to strings

        :param cx_as_list_of_dictionaries: NDEx CX data as a :py:class:`list` of
                                           :py:class:`dict` objects
        :type cx_as_list_of_dictionaries: list
        :return: 'cx_as_list_of_dictionaries' updated
        :rtype: list
        """
        LOGGER.debug('Converting values of all attributes to type string')
        for p in cx_as_list_of_dictionaries:
            k = list(p.keys())[0]
            if 'Attributes' in k:
                for i in range(len(p[k])):
                    p[k][i]['v'] = str(p[k][i]['v'])
        return cx_as_list_of_dictionaries

    @staticmethod
    def _build_post_url(time_param=None,
                        normalize_laplacian=None,
                        input_col_name=None,
                        output_prefix=None):

        params = dict()
        if time_param is not None:
            params['time'] = time_param
        if normalize_laplacian is not None and normalize_laplacian is True:
            params['normalize_laplacian'] = True
        if input_col_name is not None and input_col_name != HeatDiffusion.DEFAULT_INPUT:
            params['input_attribute_name'] = input_col_name

        if output_prefix is not None and output_prefix != HeatDiffusion.DEFAULT_OUTPUT_PREFIX:
            params['output_attribute_name'] = output_prefix

        return params

    @staticmethod
    def _append_diffusion_result_to_network(net_cx, diff_res,
                                            diffusion_output_prefix):
        """
        updates network with diffusion results

        :param net_cx:
        :param diff_res:
        :return:
        """
        if diffusion_output_prefix is None:
            o_prefix = HeatDiffusion.DEFAULT_OUTPUT_PREFIX
        else:
            o_prefix = diffusion_output_prefix
        rank_col = o_prefix + HeatDiffusion.DEFAULT_RANK_SUFFIX
        heat_col = o_prefix + HeatDiffusion.DEFAULT_HEAT_SUFFIX

        LOGGER.debug('Appending diffusion output to network')
        for aspect in diff_res['data']:
            if 'nodeAttributes' not in aspect:
                continue
            for n_attr in aspect['nodeAttributes']:
                if n_attr['n'] == rank_col or n_attr['n'] == heat_col:
                    if n_attr['d'] == 'float':
                        n_type = 'double'
                        val = float(n_attr['v'])
                    else:
                        n_type = n_attr['d']
                        val = int(n_attr['v'])
                    net_cx.add_node_attribute(property_of=int(n_attr['po']),
                                              name=n_attr['n'],
                                              values=val,
                                              type=n_type,
                                              overwrite=True)
        LOGGER.debug('Network updated')
        return net_cx

    @staticmethod
    def add_seed_nodes_by_node_name(cxnetwork, seed_nodes=None,
                                    diffusion_col_name=None):
        """
        Adds seed nodes by node name. This is done by adding a
        node attribute named by 'diffusion_col_name' :parameter
        or if that is `None` then `HeatDiffusion.DEFAULT_INPUT` is
        used

        :param cxnetwork: network to update
        :type cxnetwork: :py:class:`ndex2.nice_cx_network.NiceCXNetwork`
        :param seednodes: if this is a :py:func:`list` then the value of the node
                          attribute is set to `1.0`. If :py:func:`dict` then list of node names or dict of node names
        :type seed_nodes: list or dict
        :param diffusion_col_name:
        :return:
        """
        if cxnetwork is None:
            raise HeatDiffusionError('Input network cannot be None')

        if diffusion_col_name is None:
            diffusion_col_name = HeatDiffusion.DEFAULT_INPUT

        if isinstance(seed_nodes, list) is True:
            seed_nodes_list = seed_nodes
            is_dict = False
        else:
            if isinstance(seed_nodes, dict) is False:
                raise HeatDiffusionError('seed_nodes must be '
                                         'of type list() or dict()')
            is_dict = True
            seed_nodes_list = set(seed_nodes.keys())

        for node_id, node_obj in cxnetwork.get_nodes():
            if node_obj['n'] in seed_nodes_list:
                if is_dict is True:
                    seedval = seed_nodes[node_obj['n']]
                else:
                    seedval = 1.0
                cxnetwork.add_node_attribute(property_of=node_id,
                                             name=diffusion_col_name,
                                             values=seedval,
                                             type='double')

    @staticmethod
    def extract_diffused_subnetwork_by_rank(cx_network, max_rank=None,
                                            rank_col=None, min_heat=None,
                                            heat_col=None):
        """
        Given a network 'cx_network' where diffusion has been run. This
        method extracts a sub network containing nodes that have a
        rank equal or less then value of 'by_max_rank' as well as all edges
        connecting those nodes.

        :param cx_network: Network where diffusion has been run
        :type cx_network: :py:class:`~ndex2.nice_cx_network.NiceCXNetwork`
        :param by_max_rank: Keep nodes with rank of this or lower
        :type by_max_rank: int
        :param
        :return: cx_network with modifications applied in place
        :rtype: :py:class:`~ndex2.nice_cx_network.NiceCXNetwork`
        """
        if rank_col is None:
            rank_col = HeatDiffusion.DEFAULT_RANK
        if heat_col is None:
            heat_col = HeatDiffusion.DEFAULT_HEAT

        nodes_to_remove = set()
        for node_id, node_obj in cx_network.get_nodes():
            if max_rank is not None:
                n_attr = cx_network.get_node_attribute(node_id, rank_col)

                if n_attr is not None and int(n_attr['v']) > max_rank:
                    nodes_to_remove.add(node_id)
                    continue
            if min_heat is not None:
                n_attr = cx_network.get_node_attribute(node_id, heat_col)
                if n_attr is not None and float(n_attr['v']) < min_heat:
                    nodes_to_remove.add(node_id)

        edges_to_remove = set()
        for edge_id, edge_obj in cx_network.get_edges():
            if edge_obj['s'] in nodes_to_remove or edge_obj['t'] in nodes_to_remove:
                edges_to_remove.add(edge_id)

        for edge_id in edges_to_remove:
            e_attrib_names = set()
            e_attributes = cx_network.get_edge_attributes(edge_id)
            if e_attributes is not None:
                for e_attr in e_attributes:
                    e_attrib_names.add(e_attr['n'])
                for e_name in e_attrib_names:
                    cx_network.remove_edge_attribute(edge_id, e_name)
            cx_network.remove_edge(edge_id)
            e_attrib_names.clear()

        for node_id in nodes_to_remove:
            n_attrib_names = set()
            n_attributes = cx_network.get_node_attributes(node_id)
            if n_attributes is not None:
                for n_attr in n_attributes:
                    n_attrib_names.add(n_attr['n'])
                for n_name in n_attrib_names:
                    cx_network.remove_node_attribute(node_id, n_name)
            n_attrib_names.clear()
            cx_network.remove_node(node_id)

        cart_layout = cx_network.get_opaque_aspect('cartesianLayout')
        if cart_layout is not None:
            new_layout = []
            for node_entry in cart_layout:
                if node_entry['node'] not in nodes_to_remove:
                    new_layout.append(node_entry)
            cx_network.set_opaque_aspect('cartesianLayout', new_layout)
        return cx_network


