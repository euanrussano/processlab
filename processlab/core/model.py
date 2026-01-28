from .nodes import Node, State, Constant, Add, Multiply, Divide

class Model:
    def __init__(self):
        self.nodes = []
        self.states = []

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        node.model = self
        if isinstance(node, State):
            self.states.append(node)

    def constant(self, value:float) -> Node:
        c = Constant(value)
        self.add_node(c)
        return c

    def add(self, node1: Node, node2: Node) -> Node:
        a = Add(node1, node2)
        self.add_node(a)
        return a

    def state(self, initial_value: float = 0.0) -> State:
        s = State(initial_value)
        self.add_node(s)
        return s

    def multiply(self, node1: Node, node2: Node) -> Node:
        m = Multiply(node1, node2)
        self.add_node(m)
        return m

    def divide(self, node1: Node, node2: Node) -> Node:
        d = Divide(node1, node2)
        self.add_node(d)
        return d