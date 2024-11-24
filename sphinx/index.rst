.. FLAMO documentation master file, created by
   sphinx-quickstart on Sun Sep 22 19:51:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FLAMO
=================================

``FLAMO`` is an open-source frequecy-sampling library for audio-module optimization in ``PyTorch``.
The code can be found in `this github repository <https://github.com/gdalsanto/flamo>`_.

Installation
---------------------------

.. code-block:: bash
    
    pip install flamo


Contents 
--------------------------

.. toctree::
   :maxdepth: 2
   :caption: Differentiable Digitial Signal Processor:

   processor/dsp
   processor/system

.. toctree::
   :maxdepth: 2
   :caption: Optimization:

   optimize/trainer

.. toctree::
   :maxdepth: 2
   :caption: Functional:

   functional