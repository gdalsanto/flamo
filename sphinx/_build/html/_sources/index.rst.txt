.. FLAMO documentation master file, created by
   sphinx-quickstart on Sun Sep 22 19:51:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FLAMO
=================================
Frequency-sampling Library for Audio-Module Optimization (FLAMO) is an open-source frequecy-sampling library for audio-module optimization in `PyTorch <https://pytorch.org/>`_.
FLAMO is designed to implement and optimize differentiable linear time-invariant audio systems and is built on the frequency-sampling filter design method. 
It allows for the creation of differentiable modules that can be used stand-alone or within the computation graph of neural networks, simplifying the development of differentiable audio systems. 
It includes predefined filtering modules and auxiliary classes for constructing, training, and logging the optimized systems, all accessible through an intuitive interface.
The code can be found in `this github repository <https://github.com/gdalsanto/flamo>`_.

Installation
---------------------------
To install FlAMO via `pip <https://packaging.python.org/en/latest/key_projects/#pip>`_, on a new python virtual environment ``flamo-env``

.. code-block:: bash
    
   python3.10 -m venv .flamo-env
   source .flamo-env/bin/activate
   pip install flamo

If you are using `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_, you might need to install ``libsndfile`` manually

.. code-block:: bash

   conda create -n flamo-env python=3.10
   conda activate flamo-env
   pip install flamo
   conda install -c conda-forge libsndfile

For local installation: clone and install dependencies on a new pyton virtual environment ``flamo-env``

.. code-block:: bash

   git clone https://github.com/gdalsanto/flamo
   cd flamo
   python3.10 -m venv .flamo-env
   source .flamo-env/bin/activate
   pip install -e .

Note that it requires python>=3.10


Contents 
--------------------------
The main code is contained in **processor** and is divided in two parts: 

* ``dsp`` contains the differentiable systems (e.g., filters, delays) and basic transformations (e.g. FFT, IFFT)
* ``system`` contains classes that allow to build differentiable systems (i.e. Series, Recursion) and other utility classes (i.e. Shell).

.. toctree::
   :maxdepth: 1
   :caption: Processor:

   processor/dsp
   processor/system

The **optimize** module contains classes for handling training, validation, results logging, etc. It comes with a set of datasets and loss functions.

.. toctree::
   :maxdepth: 1
   :caption: Optimization:

   optimize/trainer
   optimize/dataset
   optimize/loss
   optimize/utils


The **functional** module is a collector of signal processing function and transformations that can be used as, e.g., parameters mappings.

.. toctree::
   :maxdepth: 1
   :caption: Functional:

   functional

The **auxiliary** module is a collector of auxiliary function and modules that showcase and expand applications of the library.

.. toctree::
   :maxdepth: 1
   :caption: Auxiliary:

   auxiliary/eq
   auxiliary/filterbank
   auxiliary/minimize
   auxiliary/reverb
   auxiliary/scattering

Reference and Contacts 
---------------------------

If you use FLAMO in your research, please cite the following conference paper:

* Gloria Dal Santo, Gian Marco De Bortoli, Karolina Prawda, Sebastian Jiro Schlecht, Vesa Välimäki "FLAMO: An Open-Source Library for Frequency-Domain Differentiable Audio Processing", in Proceedings of the *2025 International Conference on Acoustics, Speech, and Signal Processing* (ICASSP), Hyderabad, India, 2025.

For any questions or issues, please contact 

* `Gloria Dal Santo <mailto:gloria.dalsanto@aalto.fi>`_ 
* `Gian Marco De Bortoli <mailto:gian.debortoli@aalto.fi>`_

or open an issue on the `github repository <https://github.com/gdalsanto/flamo>`_.

License
---------------------------

FLAMO is licensed under the MIT License.

.. code-block:: bash

   MIT License

   Copyright (c) 2024 Gloria Dal Santo, Gian Marco De Bortoli, Sebastian Jiro Schlecht

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.