Simulation
==========

The simulation module provides the core simulation engine and integration methods.

Simulation Class
----------------

.. autoclass:: processlab.simulation.simulation.Simulation
   :members:
   :undoc-members:
   :show-inheritance:

Integrators
-----------

Integrators are responsible for advancing the state of the system over time.

Base Integrator
~~~~~~~~~~~~~~~

.. autoclass:: processlab.simulation.integrators.Integrator
   :members:
   :undoc-members:
   :show-inheritance:

Euler Integrator
~~~~~~~~~~~~~~~~

.. autoclass:: processlab.simulation.integrators.EulerIntegrator
   :members:
   :undoc-members:
   :show-inheritance:

RK4 Integrator
~~~~~~~~~~~~~~

.. autoclass:: processlab.simulation.integrators.RK4Integrator
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. autofunction:: processlab.simulation.integrators.compute_derivatives