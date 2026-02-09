from typing import TYPE_CHECKING

from processlab import Model, Simulation, MatplotlibMultiPlotRecorder

if TYPE_CHECKING:
    pass

'''
Model: Logistic growth
dxdt = rx(1-x/K)
'''

t_start = 0
t_end = 8
n = 500

m = Model()
Cv = m.constant(0.5)
A = m.constant(0.2)
qin = m.constant(0.4)
h1 = m.state(0.0)
qout = Cv.mul(h1)
der_h = qin.sub(qout).div(A)
h1.set_derivative(der_h)
h2 = m.state(0.0)
qout2 = Cv.mul(h2)
der_h2 = qout.sub(qout2).div(A)
h2.set_derivative(der_h2)

sim = Simulation(m)
sim.add_recorder(MatplotlibMultiPlotRecorder(
    label_prefix="Tank - "
))
sim.run(t_start, t_end, n)
