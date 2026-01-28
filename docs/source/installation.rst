Installation
============

Requirements
------------

ProcessLab requires:

* Python 3.8 or higher
* NumPy 2.0.0 or higher

Optional dependencies:

* matplotlib 3.5.0 or higher (for visualization)
* scipy 1.10.0 or higher (for scientific computing features)

Installing from PyPI
--------------------

The easiest way to install ProcessLab is using pip:

.. code-block:: bash

   pip install processlab

With Optional Dependencies
--------------------------

For visualization support:

.. code-block:: bash

   pip install processlab[visualization]

For scientific computing features:

.. code-block:: bash

   pip install processlab[scientific]

For all optional dependencies:

.. code-block:: bash

   pip install processlab[all]

Installing from Source
----------------------

To install the development version from GitHub:

.. code-block:: bash

   git clone https://github.com/euanrussano/processlab.git
   cd processlab
   pip install -e ".[all,dev]"

Verifying Installation
----------------------

To verify that ProcessLab is installed correctly:

.. code-block:: python

   import processlab
   print(processlab.__version__)

This should print the version number without any errors.

Virtual Environments
--------------------

It's recommended to use a virtual environment:

Using venv
~~~~~~~~~~

.. code-block:: bash

   python -m venv processlab_env
   source processlab_env/bin/activate  # On Windows: processlab_env\Scripts\activate
   pip install processlab[all]

Using conda
~~~~~~~~~~~

.. code-block:: bash

   conda create -n processlab python=3.11
   conda activate processlab
   pip install processlab[all]