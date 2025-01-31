'''implements a simple forward and backward propagation using a simple computational graph,
    on a multivariable function: f(x,y,z)=(x+y)z'''


import numpy as np

class Node:
    def __init__(self, value, grad=0.0):
        self.value = value  # Store the value of the node
        self.grad = grad  # Gradient initialized to zero
        self._backward = lambda: None  # Placeholder for backward function

    def backward(self):
        self._backward()  # Call the backward function

class ComputationalGraph:
    def __init__(self, x, y, z):
        # Convert inputs into Node objects
        self.x = Node(np.array(x, dtype=np.float32))
        self.y = Node(np.array(y, dtype=np.float32))
        self.z = Node(np.array(z, dtype=np.float32))

        # Define backward functions (for backpropagation)
        def _backward_f():
            self.q.grad += self.z.value * self.f.grad  # df/dq
            self.z.grad += self.q.value * self.f.grad  # df/dz

        def _backward_q():
            self.x.grad += 1.0 * self.q.grad  # dq/dx
            self.y.grad += 1.0 * self.q.grad  # dq/dy

        # Attach backward functions to nodes
        self._backward_f = _backward_f
        self._backward_q = _backward_q

    def forward(self):
        self.q = Node(self.x.value + self.y.value)  # q = x + y
        self.f = Node(self.q.value * self.z.value)  # f = q * z

        # Attach backward functions to nodes
        self.f._backward = self._backward_f
        self.q._backward = self._backward_q

    def backward(self):
        # Start with the final gradient (df/df = 1)
        self.f.grad = 1.0

        # Perform backpropagation
        self.f.backward()
        self.q.backward()

        return self.x.grad, self.y.grad, self.z.grad

# Example usage
x, y, z = -2, 5, -4
graph = ComputationalGraph(x, y, z)
graph.forward()
df_dx, df_dy, df_dz = graph.backward()

# Print results
print(f"Forward pass:\n\tf({x}, {y}, {z}) = {graph.f.value}")
print(f"Gradients:\n\tdf/dx = {df_dx},\n\tdf/dy = {df_dy},\n\tdf/dz = {df_dz}")