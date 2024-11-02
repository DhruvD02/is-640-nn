from engine import Value
from nn import MLP

xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]  # Input data
ys = [[0.0], [1.0], [1.0], [0.0]]  # Target outputs

nin = 2  # Number of input features
nouts = [1]  # Output features as a list
n = MLP(nin=nin, nouts=nouts)

for k in range(20):
    # Forward pass
    ypred = [n(x) for x in xs]
    # Calculate loss as a Value object
    loss = sum((Value(yout.data) - Value(ygt[0])) ** 2 for ygt, yout in zip(ys, ypred))

    for p in n.parameters():
        p.grad = 0.0
    loss.backward()  # This should now work

    for p in n.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)
