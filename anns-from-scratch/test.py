import numpy as np

from model import Model
from layers import Dense
from activations import ReLU, Tanh, Sigmoid
from losses import MSELoss

if __name__ == "__main__":
    x = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4,2,1))
    y = np.reshape([[0], [1], [1], [0]], (4,1,1))

    model = Model([Dense(2, 4), Tanh(), Dense(4, 1), Tanh()], MSELoss)

    model.fit(x, y, epochs=100, learning_rate=0.1, verbose=True)

    print(model.predict([[0], [1]]))
