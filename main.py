from dataclasses import dataclass

import matplotlib.pyplot as plt
from typing import List, Optional, TextIO, TYPE_CHECKING, Any, Sequence, Dict

from abc import ABC, abstractmethod
import csv

import numpy as np

if TYPE_CHECKING:
    from _csv import writer

class Node(ABC):
    def __init__(self) -> None:
        self.model: Optional[Model] = None
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

class SimulationRecorder(ABC):
    @abstractmethod
    def start(self, sim: 'Simulation'):
        pass

    @abstractmethod
    def record(self, sim: 'Simulation'):
        pass

    @abstractmethod
    def finalize(self, sim: 'Simulation'):
        pass


class CSVSimulationRecorder(SimulationRecorder):

    def __init__(self, filename: str):
        self.filename = filename
        self.file: Optional[TextIO] = None
        self.writer: Any = None
        self.header_written = False

    def start(self, sim: 'Simulation'):
        if self.file is None:
            self.file = open(self.filename, 'w', newline='')
            self.writer = csv.writer(self.file)

    def record(self, sim: 'Simulation'):
        if not self.header_written:
            # Write header
            header = ['time']
            for i, state in enumerate(sim.model.states):
                # You can add names to states if desired, for now use indices
                header.append(f'state_{i}')

            for detector in sim.detectors:
                header.append(f"event_{detector.tag}")

            self.writer.writerow(header)
            self.header_written = True

        # Write data row
        events = {e.tag for e in sim.events}

        row = [sim.time]
        for state in sim.model.states:
            row.append(state.value)

        for detector in sim.detectors:
            row.append(1 if detector.tag in events else 0)

        self.writer.writerow(row)

    def finalize(self, sim: 'Simulation'):
        if self.file:
            self.file.close()
            self.file = None
            self.header_written = False


class TerminalSimulationRecorder(SimulationRecorder):
    def __init__(self, print_every: int = 1):
        self.print_every = print_every
        self.step_count = 0
        self.header_printed = False

    def start(self, sim: 'Simulation'):
        pass

    def record(self, sim: 'Simulation'):
        if not self.header_printed:
            # Print header
            header = f"{'Time':<12}"
            for i, state in enumerate(sim.model.states):
                header += f"State_{i:<10}"
            print(header)
            print("-" * len(header))
            self.header_printed = True

        if self.step_count % self.print_every == 0:
            # Print data row
            row = f"{sim.time:<12.4f}"
            for state in sim.model.states:
                row += f"{state.value:<12.6f}"
            print(row)

        for event in sim.events:
            print(f"[t={event.time:.4f}] EVENT: {event.tag}")

        self.step_count += 1

    def finalize(self, sim: 'Simulation'):
        self.step_count = 0
        self.header_printed = False
        print()  # Add blank line at end


class MatplotlibRecorder(SimulationRecorder):
    def __init__(self,
                 save_path: Optional[str] = None,
                 show: bool = True,
                 plot_type: str = 'time_series',  # 'time_series' or 'phase'
                 figsize: tuple = (10, 6)):
        """
        Matplotlib recorder for simulation data

        Args:
            save_path: Path to save the figure (e.g., 'plot.png'). If None, won't save.
            show: Whether to display the plot
            plot_type: 'time_series' to plot states vs time, 'phase' for phase portrait
            figsize: Figure size as (width, height)
        """
        self.save_path = save_path
        self.show = show
        self.plot_type = plot_type
        self.figsize = figsize

        # Data storage
        self.times: List[float] = []
        self.state_values: List[List[float]] = []  # List of lists, one per state
        self.num_states: int = 0
        self.event_times: Dict[str, List[float]] = {}

    def start(self, sim: 'Simulation'):
        # Initialize state_values on first call
        self.num_states = len(sim.model.states)
        self.state_values = [[] for _ in range(self.num_states)]
        self.event_times = {}

    def record(self, sim: 'Simulation'):
        # Record data
        self.times.append(sim.time)
        for i, state in enumerate(sim.model.states):
            self.state_values[i].append(state.value)
        for event in sim.events:
            self.event_times.setdefault(event.tag, []).append(event.time)

    def finalize(self, sim: 'Simulation'):
        import matplotlib.pyplot as plt

        if not self.times:
            print("No data to plot")
            return

        if self.plot_type == 'time_series':
            self._plot_time_series(plt)
        elif self.plot_type == 'phase':
            self._plot_phase(plt)
        else:
            raise ValueError(f"Unknown plot_type: {self.plot_type}")

        for tag, times in self.event_times.items():
            for t in times:
                for ax in axes:
                    ax.axvline(t, linestyle="--", alpha=0.4, label=tag)

        # Avoid duplicate legend entries
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(dict(zip(labels, handles)).values(),
                       dict(zip(labels, handles)).keys())

        if self.save_path:
            plt.savefig(self.save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {self.save_path}")

        if self.show:
            plt.show()
        else:
            plt.close()

        # Reset data
        self.times = []
        self.state_values = [[] for _ in range(self.num_states)]

    def _plot_time_series(self, plt):
        """Plot state(s) vs time"""
        fig, axes = plt.subplots(self.num_states, 1, figsize=self.figsize, squeeze=False)

        for i in range(self.num_states):
            ax = axes[i, 0]
            ax.plot(self.times, self.state_values[i], linewidth=1.5)
            ax.set_ylabel(f'State {i}')
            ax.grid(True, alpha=0.3)

            if i == self.num_states - 1:
                ax.set_xlabel('Time')
            if i == 0:
                ax.set_title('State Variables vs Time')

        plt.tight_layout()

    def _plot_phase(self, plt):
        """Plot phase portrait (state_y vs state_x)"""
        # TODO()
        '''
        fig, ax = plt.subplots(figsize=self.figsize)

        x_data = self.state_values[self.x_state_index]
        y_data = self.state_values[self.y_state_index]

        ax.plot(x_data, y_data, linewidth=1.5)
        ax.set_xlabel(f'State {self.x_state_index}')
        ax.set_ylabel(f'State {self.y_state_index}')
        ax.set_title('Phase Portrait')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Mark start and end points
        ax.plot(x_data[0], y_data[0], 'go', markersize=8, label='Start')
        ax.plot(x_data[-1], y_data[-1], 'ro', markersize=8, label='End')
        ax.legend()

        plt.tight_layout()
        '''

class MatplotlibMultiPlotRecorder(SimulationRecorder):
    def __init__(self,
                 save_path: Optional[str] = None,
                 show: bool = True,
                 axes: Optional[Sequence[plt.Axes]] = None,
                 figsize: tuple = (15, 5),
                 label_prefix: str = ""
        ):
        """
        Matplotlib recorder that creates multiple subplots:
        - Time series for each state
        - Phase portrait

        Args:
            save_path: Path to save the figure. If None, won't save.
            show: Whether to display the plot
            axes: Optional external axes for overlaying multiple simulations
            figsize: Figure size as (width, height)
            label_prefix: Prefix added to plot labels (e.g. 'Euler', 'RK4')
        """
        self.save_path = save_path
        self.show = show
        self.axes = axes
        self.figsize = figsize
        self.label_prefix = label_prefix

        # Data storage
        self.times: List[float] = []
        self.state_values: List[List[float]] = []
        self.num_states = 0
        self._owns_figure = axes is None
        self.event_times: Dict[str, List[float]] = {}

    def start(self, sim: 'Simulation'):
        self.num_states = len(sim.model.states)
        self.state_values = [[] for _ in range(self.num_states)]
        self.event_times = {}

    def record(self, sim: 'Simulation'):
        self.times.append(sim.time)
        for i, state in enumerate(sim.model.states):
            self.state_values[i].append(state.value)
        for event in sim.events:
            self.event_times.setdefault(event.tag, []).append(event.time)

    def finalize(self, sim: 'Simulation'):

        if not self.times:
            print("No data to plot")
            return

        # Create subplots: one for each state + one for phase portrait
        num_plots = self.num_states + 1
        # Create or reuse axes
        if self.axes is None:
            fig, axes = plt.subplots(1, num_plots, figsize=self.figsize)
            # If only one plot, axes is not a list
            if num_plots == 1:
                axes = [axes]
        else:
            axes = list(self.axes)
            if len(axes) < num_plots:
                raise ValueError(
                    f"Expected at least {num_plots} axes, got {len(axes)}"
                )

        # Plot each state vs time
        for i in range(self.num_states):
            label = f"{self.label_prefix}State {i}"
            axes[i].plot(self.times, self.state_values[i], linewidth=1.5, label=label)
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(f'State {i}')
            axes[i].set_title(f'State {i} vs Time')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        for tag, times in self.event_times.items():
            for t in times:
                for ax in axes:
                    ax.axvline(t, linestyle="--", alpha=0.4, label=tag)

        # Avoid duplicate legend entries
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(dict(zip(labels, handles)).values(),
                       dict(zip(labels, handles)).keys())

        # Plot phase portrait (if at least 2 states)
        if self.num_states >= 2:
            ax_phase = axes[-1]
            x_data = self.state_values[0]
            y_data = self.state_values[1]

            label = f"{self.label_prefix}Phase"
            ax_phase.plot(x_data, y_data, linewidth=1.5, label=label)
            ax_phase.set_xlabel('State 0')
            ax_phase.set_ylabel('State 1')
            ax_phase.set_title('Phase Portrait')
            ax_phase.grid(True, alpha=0.3)
            ax_phase.axis('equal')
            ax_phase.legend()

            # Mark start and end
            ax_phase.plot(x_data[0], y_data[0], 'go', markersize=8, label='Start')
            ax_phase.plot(x_data[-1], y_data[-1], 'ro', markersize=8, label='End')
            ax_phase.legend()

        if self._owns_figure:
            plt.tight_layout()

            if self.save_path:
                plt.savefig(self.save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {self.save_path}")

            if self.show:
                plt.show()
            else:
                plt.close()

        # Reset data
        self.times = []
        self.state_values = [[] for _ in range(self.num_states)]

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

class SimulationListener(ABC):
    @abstractmethod
    def on_step(self, sim: 'Simulation'):
        pass

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

class Simulation:
    def __init__(self, model: Model, dt: float = 0.01, integrator: Integrator = EulerIntegrator()):
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

    def step(self):
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
            self.step()

        # Finalize all recorders
        for recorder in self.recorders:
            recorder.finalize(self)




m = Model()
x = m.constant(0.1)
y = m.constant(0.2)
z = x.add(y)

simulation = Simulation(m)
simulation.step()

#print(simulation.time)
print(z.value)

model = Model()
k = model.constant(-2/3600)
x = model.state(1.0)
y = k.multiply(x)
x.set_derivative(y)

# Run simulation
sim = Simulation(model, dt=0.01)  # Time step
sim.reset()
while sim.time < 8:
    sim.step()
    print(x.value)

sim.reset()
sim.run(0, 8, 500)


sim.reset()
sim.add_recorder(CSVSimulationRecorder('output.csv'))
sim.run(0, 8, 500)


sim.reset()
sim.add_recorder(TerminalSimulationRecorder())
sim.run(0, 8, 500)

logistic_growth = {
    'x0': 0.1,
    'K': 1.0,
    'r': 0.2
}
t_start = 0
t_end = 50

m = Model()
K = m.constant(1.0)
r = m.constant(0.2)
x = m.state(0.1)
der_x = r.mul(x).mul(m.constant(1.0).sub(x.div(K)))
x.set_derivative(der_x)


fig, axes = plt.subplots(1, 2, figsize=(15, 5))


sim = Simulation(m, 0.01)
sim.add_recorder(MatplotlibMultiPlotRecorder(
    axes=axes,
    label_prefix="Euler - "
))
sim.add_recorder(CSVSimulationRecorder("logistic_growth.csv"))
sim.run(t_start, t_end, 500)

sim.reset()
sim.run(t_start, t_end, 10)

sim2 = Simulation(m, 0.01, integrator=RK4Integrator())
sim2.add_recorder(MatplotlibMultiPlotRecorder(
    axes=axes,
    label_prefix="RK4 - "
))
sim2.run(t_start, t_end, 500)
sim2.reset()
sim2.run(t_start, t_end, 10)

sim.reset()
sim.add_detector(ThresholdDetector(0, 0.8))
sim.run(t_start, t_end, 100)



import math
from typing import List

def logistic_growth_analytical(x0: float, r: float, K: float, time_points: Sequence[float]) -> Sequence[float]:
    """
    Analytical solution of: dx/dt = r*x*(1 - x/K), with x(0) = x0

    x(t) = K / (1 + ((K - x0)/x0) * exp(-r t))
    """
    if K == 0:
        raise ValueError("K must be non-zero")

    # Edge cases
    if x0 == 0.0:
        return [0.0 for _ in time_points]
    if x0 == K:
        return [K for _ in time_points]

    a = (K - x0) / x0  # ((K - x0)/x0)

    xs: List[float] = []
    for t in time_points:
        xs.append(K / (1.0 + a * math.exp(-r * t)))

    return xs


ts = np.linspace(t_start, t_end, 100).tolist()
xs = logistic_growth_analytical(logistic_growth['x0'], logistic_growth['r'], logistic_growth['K'], ts)

axes[0].plot(ts, xs, label = "Analytical")
axes[0].legend()

plt.tight_layout()
plt.show()

class StabilitySimulationListener(SimulationListener):
    def __init__(self, tolerance: float = 1e-3):
        self.x = None
        self.tolerance = tolerance
        self.n_tries = 0

    def on_step(self, sim: 'Simulation'):
        new_x = sim.model.states[0].value
        t = sim.time

        dx = 100*self.tolerance
        if self.x:
            dx = abs(self.x - new_x)

        self.x = new_x

        # Example thresholds
        if dx < self.tolerance:
            self.n_tries += 1
        else:
            self.n_tries = 0

        if self.n_tries == 3:
            print(f"[t={t:.2f}] State reached")




sim.clear_recorders()
sim.reset()
sim.listener = StabilitySimulationListener()
sim.run(t_start, t_end, 100)


