import csv
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, TextIO, Any, List, Dict, Sequence

from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from .simulation import Simulation


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