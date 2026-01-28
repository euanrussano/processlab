from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

from .integrators import EulerIntegrator

if TYPE_CHECKING:
    from processlab.core import Model
    from .recorders import SimulationRecorder
    from .integrators import Integrator, EulerIntegrator
    from .events import EventDetector, SimulationEvent
    from .listener import SimulationListener


class Simulation:
    def __init__(self, model: 'Model', dt: float = 0.01, integrator: Integrator = EulerIntegrator()):
        self.model = model
        self.dt = dt
        self.time = 0.0
        self.recorders: List[SimulationRecorder] = []
        self.integrator = integrator
        self.listener: Optional[SimulationListener] = None
        self.detectors: List[EventDetector] = []
        self.events: list[SimulationEvent] = []

    def add_detector(self, detector: EventDetector):
        self.detectors.append(detector)

    def add_recorder(self, recorder: SimulationRecorder):
        self.recorders.append(recorder)

    def clear_recorders(self):
        self.recorders = []

    def __step(self):
        # First, update all non-state nodes
        for node in self.model.nodes:
            inputs = []
            for input_node in node.inputs:
                inputs.append(input_node.value)
            new_value = node.update(inputs)
            node.value = new_value

        # Then, integrate state variables
        self.integrator.step(self)

        self.time += self.dt

        for detector in self.detectors:
            if not detector.triggered and detector.detect(self):
                detector.triggered = True
                event = detector.emit(self)
                self.events.append(event)

        # evoke listener callback
        if self.listener:
            self.listener.on_step(self)

        # Record after each step
        for recorder in self.recorders:
            recorder.record(self)

    def reset(self):
        for state_node in self.model.states:
            state_node.reset()
        self.events = []

    def run(self, start: float, end: float, steps: int, cold_start: bool = True):
        if cold_start:
            self.reset()
        self.dt = (end - start) / steps
        self.time = start

        for recorder in self.recorders:
            recorder.start(self)

        # ✅ Record initial condition
        for recorder in self.recorders:
            recorder.record(self)

        while self.time < end:
            self.__step()

        # Finalize all recorders
        for recorder in self.recorders:
            recorder.finalize(self)

