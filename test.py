from . import Optimizers
from Net import NeuronalNet
from Layer import Linear_Layer
from LossFunctions import MSE_Loss
from Optimizer import Optim
import numpy as np

Victor = NeuronalNet([
    Linear_Layer((1,1), [np.matrix([[2.0,1.0], [-1.0, 0.0]])]),
    MSE_Loss()
])

optimizer = Optimizers.Momentum(Victor, 0.01, 0.5)

feature = np.matrix([[1]])
label = np.matrix([[1]])

Victor.layers[0].gradient = [np.matrix([[0.5, 1.0], [4.0, 1.0]])]

optimizer.step()

Victor.layers[0].gradient = [np.matrix([[4.0, -2.0], [-2.0, 1.0]])]

optimizer.step()

print(Victor.layers[0].weights)




