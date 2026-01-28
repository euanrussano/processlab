from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import Simulation

class SimulationListener(ABC):
    @abstractmethod
    def on_step(self, sim: 'Simulation'):
        pass
