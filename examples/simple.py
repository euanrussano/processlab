from processlab import Model, Simulation

'''
z = x + y
where x = 0.1 , y = 0.2
i.e z = 0.3
'''
m = Model()
x = m.constant(0.1)
y = m.constant(0.2)
z = x.add(y)

simulation = Simulation(m)
simulation.step()

print(z.value)