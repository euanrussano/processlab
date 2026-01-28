Core Components
===============

This module contains the core building blocks of ProcessLab: nodes and models.

Nodes
-----

Nodes are the fundamental units of computation in ProcessLab. They form a computational
graph where values flow from inputs through operations to outputs.

Base Node Class
~~~~~~~~~~~~~~~

.. autoclass:: processlab.core.nodes.Node
   :members:
   :undoc-members:
   :show-inheritance:

Node Types
~~~~~~~~~~

.. autoclass:: processlab.core.nodes.Constant
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: processlab.core.nodes.Add
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: processlab.core.nodes.Multiply
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: processlab.core.nodes.Divide
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: processlab.core.nodes.State
   :members:
   :undoc-members:
   :show-inheritance:

Model
-----

The Model class manages the computational graph and provides convenient methods
for creating nodes.

.. autoclass:: processlab.core.model.Model
   :members:
   :undoc-members:
   :show-inheritance: