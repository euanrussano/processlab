from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, List

if TYPE_CHECKING:
    from .model import Model

class Node(ABC):
    def __init__(self) -> None:
        self.model: Optional['Model'] = None
        self.inputs: List[Node] = []
        self.value = 0.0

    @abstractmethod
    def update(self, x: List[float]) -> float:
        pass

    def add(self, other: 'Node') -> 'Node':
        if self.model:
            return self.model.add(self, other)
        else:
            raise Exception("Undefined model")

    def sub(self, other: 'Node') -> 'Node':
        if self.model:
            inv = self.model.constant(-1)
            node2 = other.mul(inv)
            return self.add(node2)
        else:
            raise Exception("Undefined model")


    def multiply(self, other: 'Node') -> 'Node':
        if self.model:
            return self.model.multiply(self, other)
        else:
            raise Exception("Undefined model")

    def mul(self, other: 'Node') -> 'Node':
        return self.multiply(other)

    def div(self, other: 'Node') -> 'Node':
        if self.model:
            return self.model.divide(self, other)
        else:
            raise Exception("Undefined model")



class Constant(Node):

    def __init__(self, value):
        super().__init__()
        self.constant_value = value

    def update(self, x: List[float]) -> float:
        return self.constant_value

class Add(Node):

    def __init__(self, node1: Node, node2: Node):
        super().__init__()
        self.inputs = [node1, node2]

    def update(self, x: List[float]):
        return sum(x)

class Multiply(Node):
    def __init__(self, node1: Node, node2: Node):
        super().__init__()
        self.inputs = [node1, node2]

    def update(self, x: List[float]):
        return x[0] * x[1]

class Divide(Node):
    def __init__(self, node1: Node, node2: Node):
        super().__init__()
        self.inputs = [node1, node2]

    def update(self, x: List[float]):
        return x[0] / x[1]

class State(Node):
    """A state variable that can be integrated over time"""

    def __init__(self, initial_value: float = 0.0):
        super().__init__()
        self.initial_value = initial_value
        self.value = initial_value
        self.derivative: Optional[Node] = None

    def update(self, x: List[float]) -> float:
        # State nodes don't update from inputs in the normal way
        # They are updated by the integrator
        return self.value

    def set_derivative(self, derivative_node: Node):
        self.derivative = derivative_node

    def reset(self):
        self.value = self.initial_value