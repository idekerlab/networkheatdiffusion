===============================
Network Heat Diffusion
===============================

.. image:: https://img.shields.io/pypi/v/networkheatdiffusion.svg
        :target: https://pypi.python.org/pypi/networkheatdiffusion

.. image:: https://img.shields.io/travis/idekerlab/networkheatdiffusion.svg
        :target: https://travis-ci.org/idekerlab/networkheatdiffusion

.. image:: https://readthedocs.org/projects/networkheatdiffusion/badge/?version=latest
        :target: https://networkheatdiffusion.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Overview
-------------


Finds network network neighborhoods in a larger
network relevant to an initial set of nodes of interest. It works by
propagating the node set across the network in a process analogous
to heat diffusing across a conductive medium. A typical application
would be discovering network mechanisms from hits in a screen or
differential expression analysis. More generally this package is
applicable for parsing a larger network to a smaller, more manageable
one based on known nodes of interest.

This code was extracted from
the `heat-diffusion cxmate service <https://github.com/idekerlab/heat-diffusion>`__


If you use this code or the `heat-diffusion cxmate service <https://github.com/idekerlab/heat-diffusion>`__
please cite the following publication:

Carlin DE, Demchak B, Pratt D, Sage E, Ideker T (2017)
Network propagation in the cytoscape cyberinfrastructure.
PLOS Computational Biology 13(10): e1005598.
https://doi.org/10.1371/journal.pcbi.1005598

Dependencies
--------------

* `ndex2 <https://pypi.org/project/ndex2>`__
* `networkx <https://pypi.org/project/networkx>`__
* `requests <https://pypi.org/project/requests>`__
* `scipy <https://pypi.org/project/scipy>`__
* `numpy <https://pypi.org/project/numpy>`__

**Compatibility**
-----------------------

Python 3.6+

.. note::

    Python 3.7+ is preferred

**Installation**
--------------------------------------

.. code-block::

    # if git not installed, repo can be downloaded and unzipped
    git clone https://github.com/idekerlab/networkheatdiffusion

    cd networkheatdiffusion
    python setup.py bdist_wheel
    pip install dist/networkheatdiffusion*whl


`Pip <https://pypi.org/project/pip>`__ install coming soon...

**License**
--------------------------------------

See `LICENSE <https://github.com/idekerlab/networkheatdiffusion/blob/master/LICENSE>`__

Credits
---------

The template for this package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

The code for this package was extracted from `heat-diffusion cxmate service <https://github.com/idekerlab/heat-diffusion>`__

If used please cite the following publication:

Carlin DE, Demchak B, Pratt D, Sage E, Ideker T (2017)
Network propagation in the cytoscape cyberinfrastructure.
PLOS Computational Biology 13(10): e1005598. https://doi.org/10.1371/journal.pcbi.1005598

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
