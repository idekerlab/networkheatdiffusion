=======
History
=======

0.4.0 (2021-10-14)
-------------------

* **Breaking change**, passing ``None`` or a network lacking nodes and or edges into
  ``HeatDiffusion.run_diffusion()`` will now raise a ``HeatDiffusionError``

0.3.0 (2021-7-27)
-------------------

* **Breaking change**, moved constants originally in ``HeatDiffusion`` class
  into separate ``constants`` module

* Improved documentation

0.2.0 (2020-08-28)
-------------------

* When `correct_rank` is set to `True` in `run_diffusion()`
  method and there are identical ranks, the subsequent rank value
  will be the next value as if all previous ranks were different.
  For example, in version `0.1.0` the rank would have been 1, 2, 2, 3 and now
  it is 1, 2, 2, 4.


0.1.0 (2020-06-30)
------------------

* First release on PyPI.
