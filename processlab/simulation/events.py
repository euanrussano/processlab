from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import Simulation

@dataclass(frozen=True)
class SimulationEvent:
    time: float
    tag: str


class EventDetector(ABC):
    def __init__(self, tag: str):
        self.tag = tag
        self.triggered = False

    @abstractmethod
    def detect(self, sim: "Simulation") -> bool:
        """
        Return True when the event is detected.
        """
        pass


    def emit(self, sim: "Simulation") -> SimulationEvent:
        return SimulationEvent(time=sim.time, tag=self.tag)

class SteadyStateDetector(EventDetector):
    def __init__(self, state_index: int, tol: float = 1e-6):
        super().__init__("steady_state")
        self.state_index = state_index
        self.tol = tol

    def detect(self, sim: 'Simulation') -> bool:
        state = sim.model.states[self.state_index]
        dxdt = state.derivative.value
        return abs(dxdt) < self.tol

class ThresholdDetector(EventDetector):
    def __init__(self, state_index: int, threshold: float):
        super().__init__(f"threshold_{threshold}")
        self.state_index = state_index
        self.threshold = threshold

    def detect(self, sim: 'Simulation') -> bool:
        return sim.model.states[self.state_index].value >= self.threshold

