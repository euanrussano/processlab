"""ProcessLab: A dynamic systems simulation library"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Euan Russano"
__email__ = "your.email@example.com"

# Core imports
from processlab.core.nodes import Node, Constant, Add, Multiply, Divide, State
from processlab.core.model import Model

# Simulation imports
from processlab.simulation.simulation import Simulation
from processlab.simulation.integrators import (
    Integrator,
    EulerIntegrator,
    RK4Integrator,
)
from processlab.simulation.events import (
    SimulationEvent,
    EventDetector,
    SteadyStateDetector,
    ThresholdDetector
)
from processlab.simulation.listener import SimulationListener

# Recorder imports
from processlab.simulation.recorders import SimulationRecorder
from processlab.simulation.recorders import CSVSimulationRecorder
from processlab.simulation.recorders import TerminalSimulationRecorder

# Optional imports
try:
    from processlab.simulation.recorders import (
        MatplotlibRecorder,
        MatplotlibMultiPlotRecorder,
    )
except ImportError:
    pass

__all__ = [
    # Core
    "Node",
    "Constant",
    "Add",
    "Multiply",
    "Divide",
    "State",
    "Model",
    # Simulation
    "Simulation",
    "Integrator",
    "EulerIntegrator",
    "RK4Integrator",
    # Events
    "SimulationEvent",
    "EventDetector",
    "SteadyStateDetector",
    "ThresholdDetector",
    "SimulationListener",
    # Recorders
    "SimulationRecorder",
    "CSVSimulationRecorder",
    "TerminalSimulationRecorder",
]