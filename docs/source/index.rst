ProcessLab Documentation
========================

.. image:: https://badge.fury.io/py/processlab.svg
   :target: https://badge.fury.io/py/processlab
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/processlab.svg
   :target: https://pypi.org/project/processlab/
   :alt: Python versions

ProcessLab is a flexible and intuitive Python library for simulating dynamic systems using computational graphs.

Features
--------

* 🔢 **Node-based computational graph** for building dynamic models
* 🧮 **Multiple integration methods** (Euler, RK4, and extensible)
* 📊 **Flexible recording system** (CSV, terminal, matplotlib)
* 🎯 **Event detection** for threshold crossing and steady states
* 🔬 **Scientific computing** support with NumPy integration
* 📈 **Visualization** with matplotlib integration

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install processlab

   # With visualization support
   pip install processlab[visualization]

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   from processlab import Model, Simulation, CSVSimulationRecorder

   # Create a simple exponential decay model: dx/dt = -k*x
   model = Model()
   k = model.constant(-0.5)
   x = model.state(1.0)  # Initial value
   x.set_derivative(k.multiply(x))

   # Run simulation
   sim = Simulation(model)
   sim.add_recorder(CSVSimulationRecorder('output.csv'))
   sim.run(start=0, end=10, steps=100)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/simulation
   api/recorders
   api/events

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   changelog
   contributing
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`