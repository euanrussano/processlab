# ProcessLab

A flexible and intuitive Python library for simulating dynamic systems using computational graphs.

## Features

- 🔢 **Node-based computational graph** for building dynamic models
- 🧮 **Multiple integration methods** (Euler, RK4, and extensible)
- 📊 **Flexible recording system** (CSV, terminal, matplotlib)
- 🎯 **Event detection** for threshold crossing and steady states
- 🔬 **Scientific computing** support with NumPy integration
- 📈 **Visualization** with matplotlib integration

## Installation

```bash
# Basic installation
pip install processlab

# With visualization support
pip install processlab[visualization]

# With all features
pip install processlab[all]
```

## Quick Start

```python
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
```

## Example: Logistic Growth

```python
from processlab import Model, Simulation, EulerIntegrator, RK4Integrator
import matplotlib.pyplot as plt

# Create logistic growth model: dx/dt = r*x*(1 - x/K)
model = Model()
r = model.constant(0.2)   # Growth rate
K = model.constant(1.0)   # Carrying capacity
x = model.state(0.1)      # Initial population

# Build derivative: r*x*(1 - x/K)
one = model.constant(1.0)
derivative = r.multiply(x).multiply(one.sub(x.div(K)))
x.set_derivative(derivative)

# Compare integrators
sim1 = Simulation(model, dt=0.01, integrator=EulerIntegrator())
sim1.run(0, 50, 500)

sim2 = Simulation(model, dt=0.01, integrator=RK4Integrator())
sim2.run(0, 50, 500)
```

## Core Concepts

### Models and Nodes

ProcessLab uses a computational graph approach where:
- **Nodes** represent values and operations
- **States** are variables that change over time
- **Models** connect nodes together

```python
from processlab import Model
model = Model()

# Create nodes
a = model.constant(2.0)
b = model.constant(3.0)
c = a.add(b)  # c = a + b

# Create state variables
x = model.state(initial_value=1.0)
```

### Integration Methods

Choose from multiple numerical integration methods:

```python
from processlab import Simulation
from processlab import EulerIntegrator, RK4Integrator

# First-order Euler method (fast, less accurate)
sim = Simulation(model, integrator=EulerIntegrator())

# Fourth-order Runge-Kutta (slower, more accurate)
sim = Simulation(model, integrator=RK4Integrator())
```

### Event Detection

Detect when specific conditions are met during simulation:

```python
from processlab import ThresholdDetector, SteadyStateDetector

# Detect when state crosses threshold
sim.add_detector(ThresholdDetector(state_index=0, threshold=0.8))

# Detect steady state
sim.add_detector(SteadyStateDetector(state_index=0, tol=1e-6))

sim.run(0, 100, 1000)

# Check detected events
for event in sim.events:
    print(f"Event '{event.tag}' at t={event.time}")
```

### Recording Results

Multiple ways to record and visualize results:

```python
from processlab import (
    CSVSimulationRecorder,
    TerminalSimulationRecorder,
    MatplotlibRecorder
)

# Save to CSV
sim.add_recorder(CSVSimulationRecorder('results.csv'))

# Print to terminal
sim.add_recorder(TerminalSimulationRecorder(print_every=10))

# Plot with matplotlib
sim.add_recorder(MatplotlibRecorder(save_path='plot.png', show=True))

sim.run(0, 10, 100)
```

## Documentation

For more examples and detailed API documentation, visit the [documentation](https://github.com/euanrussano/processlab).

## Development

```bash
# Clone repository
git clone https://github.com/euanrussano/processlab.git
cd processlab

# Install in development mode
pip install -e ".[all,dev]"

# Run tests
pytest

# Format code
black processlab/

# Type checking
mypy processlab/
```

## Requirements

- Python >= 3.8
- NumPy >= 2.0.0
- matplotlib >= 3.5.0 (optional, for visualization)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- **Euan Russano** - [GitHub](https://github.com/euanrussano)

## Acknowledgments

Built with Python and the scientific Python ecosystem (NumPy, matplotlib).