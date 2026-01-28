from typing import TYPE_CHECKING, Sequence, List

import matplotlib.pyplot as plt
import numpy as np
import math

from processlab import Model, Simulation, CSVSimulationRecorder, MatplotlibMultiPlotRecorder, ThresholdDetector, RK4Integrator, SimulationListener

if TYPE_CHECKING:
    pass

'''
Model: Logistic growth
dxdt = rx(1-x/K)
'''

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