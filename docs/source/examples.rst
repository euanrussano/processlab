Examples
========

This page contains complete examples demonstrating various features of ProcessLab.

Exponential Decay
-----------------

A simple first-order differential equation: :math:`\frac{dx}{dt} = -kx`

.. code-block:: python

   from processlab import Model, Simulation, MatplotlibRecorder

   # Create model
   model = Model()
   k = model.constant(-2.0 / 3600)  # Decay constant
   x = model.state(1.0)              # Initial value
   x.set_derivative(k.multiply(x))

   # Simulate and visualize
   sim = Simulation(model, dt=0.01)
   sim.add_recorder(MatplotlibRecorder(show=True))
   sim.run(0, 8, 500)

Logistic Growth
---------------

Population dynamics with carrying capacity: :math:`\frac{dx}{dt} = rx(1 - \frac{x}{K})`

.. code-block:: python

   from processlab import Model, Simulation, RK4Integrator
   import matplotlib.pyplot as plt

   # Model parameters
   model = Model()
   r = model.constant(0.2)   # Growth rate
   K = model.constant(1.0)   # Carrying capacity
   x = model.state(0.1)      # Initial population

   # Build derivative
   one = model.constant(1.0)
   derivative = r.multiply(x).multiply(one.sub(x.div(K)))
   x.set_derivative(derivative)

   # Compare integrators
   fig, axes = plt.subplots(1, 2, figsize=(15, 5))

   # Euler method
   sim1 = Simulation(model, dt=0.01)
   # ... add recorder and run ...

   # RK4 method (more accurate)
   sim2 = Simulation(model, dt=0.01, integrator=RK4Integrator())
   # ... add recorder and run ...

   plt.show()

Comparing Integration Methods
------------------------------

Demonstrates the difference in accuracy between Euler and RK4 methods:

.. code-block:: python

   from processlab import Model, Simulation, EulerIntegrator, RK4Integrator
   import numpy as np
   import matplotlib.pyplot as plt

   # Analytical solution for dx/dt = -kx is x(t) = x0 * exp(-kt)
   def analytical_solution(x0, k, t):
       return x0 * np.exp(-k * t)

   # Create model
   model = Model()
   k = model.constant(-0.5)
   x = model.state(1.0)
   x.set_derivative(k.multiply(x))

   # Simulate with different methods
   times = np.linspace(0, 10, 100)

   # Euler
   sim_euler = Simulation(model, integrator=EulerIntegrator())
   # ... run and record ...

   # RK4
   sim_rk4 = Simulation(model, integrator=RK4Integrator())
   # ... run and record ...

   # Analytical
   analytical = analytical_solution(1.0, 0.5, times)

   # Plot comparison
   plt.figure(figsize=(10, 6))
   # ... plot results ...
   plt.show()

Event Detection Example
-----------------------

Detect when population reaches 80% of carrying capacity:

.. code-block:: python

   from processlab import (
       Model, Simulation, ThresholdDetector,
       CSVSimulationRecorder
   )

   # Logistic growth model
   model = Model()
   r = model.constant(0.2)
   K = model.constant(1.0)
   x = model.state(0.1)

   one = model.constant(1.0)
   x.set_derivative(r.multiply(x).multiply(one.sub(x.div(K))))

   # Create simulation with detector
   sim = Simulation(model)
   sim.add_detector(ThresholdDetector(state_index=0, threshold=0.8))
   sim.add_recorder(CSVSimulationRecorder('logistic_with_events.csv'))

   sim.run(0, 50, 500)

   # Print detected events
   for event in sim.events:
       print(f"Reached {event.tag} at t={event.time:.2f}")

Custom Simulation Listener
---------------------------

Create a custom listener to monitor simulation progress:

.. code-block:: python

   from processlab import SimulationListener, Simulation, Model

   class StabilityListener(SimulationListener):
       def __init__(self, tolerance=1e-3):
           self.tolerance = tolerance
           self.previous_value = None
           self.stable_count = 0

       def on_step(self, sim):
           current_value = sim.model.states[0].value

           if self.previous_value is not None:
               change = abs(current_value - self.previous_value)

               if change < self.tolerance:
                   self.stable_count += 1
                   if self.stable_count >= 3:
                       print(f"System stabilized at t={sim.time:.2f}")
               else:
                   self.stable_count = 0

           self.previous_value = current_value

   # Use the listener
   model = Model()
   # ... build model ...

   sim = Simulation(model)
   sim.listener = StabilityListener(tolerance=1e-3)
   sim.run(0, 100, 1000)

Multi-State Systems
--------------------

Simulate systems with multiple coupled states:

.. code-block:: python

   from processlab import Model, Simulation

   # Predator-prey model (Lotka-Volterra)
   # dx/dt = ax - bxy  (prey)
   # dy/dt = -cy + dxy (predator)

   model = Model()

   # Parameters
   a = model.constant(1.0)   # Prey growth rate
   b = model.constant(0.1)   # Predation rate
   c = model.constant(1.5)   # Predator death rate
   d = model.constant(0.075) # Predator efficiency

   # States
   x = model.state(10.0)  # Initial prey
   y = model.state(5.0)   # Initial predators

   # Derivatives
   dx_dt = a.multiply(x).sub(b.multiply(x).multiply(y))
   dy_dt = d.multiply(x).multiply(y).sub(c.multiply(y))

   x.set_derivative(dx_dt)
   y.set_derivative(dy_dt)

   # Simulate
   sim = Simulation(model, dt=0.01)
   sim.run(0, 50, 5000)