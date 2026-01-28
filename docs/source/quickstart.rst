Quick Start Guide
=================

This guide will get you up and running with ProcessLab in minutes.

Your First Simulation
---------------------

Let's create a simple exponential decay model: :math:`\frac{dx}{dt} = -kx`

.. code-block:: python

   from processlab import Model, Simulation

   # Create a model
   model = Model()

   # Define the decay constant
   k = model.constant(-0.5)

   # Create a state variable with initial value
   x = model.state(1.0)

   # Set the derivative: dx/dt = k * x
   x.set_derivative(k.multiply(x))

   # Create and run simulation
   sim = Simulation(model, dt=0.01)
   sim.run(start=0, end=10, steps=100)

   # Check final value
   print(f"Final value: {x.value}")

Recording Data
--------------

To save simulation results:

CSV Output
~~~~~~~~~~

.. code-block:: python

   from processlab import Model, Simulation, CSVSimulationRecorder

   model = Model()
   k = model.constant(-0.5)
   x = model.state(1.0)
   x.set_derivative(k.multiply(x))

   sim = Simulation(model)
   sim.add_recorder(CSVSimulationRecorder('output.csv'))
   sim.run(0, 10, 100)

Terminal Output
~~~~~~~~~~~~~~~

.. code-block:: python

   from processlab import TerminalSimulationRecorder

   sim = Simulation(model)
   sim.add_recorder(TerminalSimulationRecorder(print_every=10))
   sim.run(0, 10, 100)

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   from processlab import MatplotlibRecorder

   sim = Simulation(model)
   sim.add_recorder(MatplotlibRecorder(
       save_path='plot.png',
       show=True
   ))
   sim.run(0, 10, 100)

Using Different Integrators
----------------------------

ProcessLab supports multiple integration methods:

.. code-block:: python

   from processlab import Simulation, EulerIntegrator, RK4Integrator

   # Euler method (fast, less accurate)
   sim_euler = Simulation(model, integrator=EulerIntegrator())

   # Runge-Kutta 4th order (slower, more accurate)
   sim_rk4 = Simulation(model, integrator=RK4Integrator())

Building Complex Models
-----------------------

Let's create a logistic growth model: :math:`\frac{dx}{dt} = rx(1 - \frac{x}{K})`

.. code-block:: python

   from processlab import Model, Simulation

   model = Model()

   # Parameters
   r = model.constant(0.2)   # Growth rate
   K = model.constant(1.0)   # Carrying capacity

   # State variable
   x = model.state(0.1)      # Initial population

   # Build derivative: r * x * (1 - x/K)
   one = model.constant(1.0)
   term1 = r.multiply(x)
   term2 = one.sub(x.div(K))
   derivative = term1.multiply(term2)

   x.set_derivative(derivative)

   # Simulate
   sim = Simulation(model, dt=0.01)
   sim.run(0, 50, 500)

Event Detection
---------------

Detect when specific conditions are met:

.. code-block:: python

   from processlab import ThresholdDetector, SteadyStateDetector

   model = Model()
   # ... build your model ...

   sim = Simulation(model)

   # Detect when state crosses threshold
   sim.add_detector(ThresholdDetector(state_index=0, threshold=0.8))

   # Detect steady state
   sim.add_detector(SteadyStateDetector(state_index=0, tol=1e-6))

   sim.run(0, 100, 1000)

   # Check detected events
   for event in sim.events:
       print(f"Event '{event.tag}' detected at t={event.time:.2f}")

Next Steps
----------

* Explore the :doc:`examples` for more complex use cases
* Read the :doc:`api/core` for detailed API documentation
* Check out the GitHub repository for more examples