from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import Simulation
    from processlab.core.nodes import State

class Integrator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, sim: "Simulation"):
        pass

def compute_derivatives(sim: "Simulation") -> dict[State, float]:
    derivatives = {}
    for state in sim.model.states:
        if not state.derivative:
            continue

        inputs = [n.value for n in state.derivative.inputs]
        derivatives[state] = state.derivative.update(inputs)

    return derivatives

class EulerIntegrator(Integrator):
    def __init__(self):
        super().__init__()

    def step(self, sim: "Simulation"):
        dt = sim.dt

        derivatives = compute_derivatives(sim)
        for state in sim.model.states:
            if not state.derivative:
                continue

            dxdt = derivatives[state]
            state.value += dxdt * dt

class RK4Integrator(Integrator):
    def __init__(self):
        super().__init__()



    def step(self, sim: "Simulation"):
        dt = sim.dt
        states = sim.model.states

        # Save original values
        original = {s: s.value for s in states}

        # k1
        k1 = compute_derivatives(sim)

        # k2
        for s in k1:
            s.value = original[s] + 0.5 * dt * k1[s]
        k2 = compute_derivatives(sim)

        # k3
        for s in k2:
            s.value = original[s] + 0.5 * dt * k2[s]
        k3 = compute_derivatives(sim)

        # k4
        for s in k3:
            s.value = original[s] + dt * k3[s]
        k4 = compute_derivatives(sim)

        # Final update
        for s in original:
            s.value = (
                    original[s]
                    + (dt / 6.0)
                    * (k1[s] + 2 * k2[s] + 2 * k3[s] + k4[s])
            )