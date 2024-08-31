Installation
============

There is currently no support for PyPi or conda, so you (unfortunately) need to install from source.
To install the package, run the following command:

.. code-block:: bash
    pip install git+https://github.com/DhayaaAnbajagane/Baryonification.git

or alternatively you can download the repo yourself and set it up,

.. code-block:: bash
    git clone https://github.com/DhayaaAnbajagane/Baryonification.git
    cd Baryonification
    pip install -e .

This will keep the source files in the location you git clone'd from.

There are known package *curiosities* in the current setup. To simplify things, the root
directory contains a `environment.yml <https://github.com/DhayaaAnbajagane/Baryonification/blob/main/environment.yaml>`_ file that can be used to construct an environment.
The main issue is in getting a version of CCL where the profiles pickle properly. The
pickling is needed for running all Parallelization routines. I got it to work for
specific version of joblib/ccl, so that is what is included in `environment.yml <https://github.com/DhayaaAnbajagane/Baryonification/blob/main/environment.yaml>`_ file.
