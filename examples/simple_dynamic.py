from processlab import Model, Simulation
from processlab.simulation.recorders import CSVSimulationRecorder, TerminalSimulationRecorder

'''
Model: 
y = (-2/3600)*x
dxdt = y
'''
model = Model()
k = model.constant(-2/3600)
x = model.state(1.0)
y = k.multiply(x)
x.set_derivative(y)

# Run simulation
sim = Simulation(model, dt=0.01)  # Time step
sim.run(0, 8, 500)


sim.reset()
sim.add_recorder(CSVSimulationRecorder('simple_dynamic.csv'))
sim.run(0, 8, 500)


sim.reset()
sim.add_recorder(TerminalSimulationRecorder())
sim.run(0, 8, 500)