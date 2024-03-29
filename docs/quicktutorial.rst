================
Quick Tutorial
================

To use Network Heat Diffusion in a project::

    import json
    import ndex2
    import networkheatdiffusion

    # Create NDEx2 python client
    client = ndex2.client.Ndex2()

    # load Bioplex 2.0 (~56,000 interactions) network from NDEx
    client_resp = client.get_network_as_cx_stream('98ba6a19-586e-11e7-8f50-0ac135e8bacf')

    # Convert downloaded network to NiceCXNetwork object
    net_cx = ndex2.create_nice_cx_from_raw_cx(json.loads(client_resp.content))

    # create diffusion object
    diffuser = networkheatdiffusion.HeatDiffusion()

    # set seed nodes
    diffuser.add_seed_nodes_by_node_name(net_cx, seed_nodes=['NOTCH1', 'NOTCH2',
                                                             'NOTCH3'])

    diffused_network = diffuser.run_diffusion(net_cx)

    print('Node,Rank,Heat')
    for node_id, node_obj in diffused_network.get_nodes():
        node_heat = diffused_network.get_node_attribute(node_id, 'diffusion_output_heat')
        if node_heat is None or node_heat == (None, None):
            continue
        node_rank = diffused_network.get_node_attribute(node_id, 'diffusion_output_rank')
        if node_rank is None or node_rank == (None, None):
            continue
        print(node_obj['n'] + ',' + str(node_rank['v']) + ',' + str(node_heat['v']))








The above adds ``diffusion_output_heat`` and ``diffusion_output_rank`` attributes to
nodes on network that can be used to filtered to create a sub network.
