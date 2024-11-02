from engine import Value
from nn import MLP

# Sample data (input features and target outputs)
xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]  # Input data
ys = [[0.0], [1.0], [1.0], [0.0]]  # Target outputs

# Initialize the MLP with input and output sizes
nin = 2  # Number of input features
nouts = [1]  # Output features as a list
n = MLP(nin=nin, nouts=nouts)

# Training loop
for k in range(20):
    # Forward pass
    ypred = [n(x) for x in xs]
    # Calculate loss as a Value object
    loss = sum((Value(yout.data) - Value(ygt[0])) ** 2 for ygt, yout in zip(ys, ypred))

    # Backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()  # This should now work

    # Update weights
    for p in n.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)
